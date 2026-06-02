import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tfd

from torch_wm.core.registry import ModuleRegistry
from torch_wm.structs import AttrDict
from torch_wm import modules
from torch_wm import distributions

@ModuleRegistry.register('rssm')
class RSSM(nn.Module):
    def __init__(
        self,
        num_actions,
        stoch_size=32,
        discrete=32,
        hidden_size=1024,
        unroll=False,
        learn_initial=True,
        uniform_mix=0.01,
        action_clip=1.0,
        weight_init="dreamerv3_normal",
        bias_init="zeros",
        dist_weight_init="xavier_uniform",
        dist_bias_init="zeros",
        **kw,
    ):
        super(RSSM, self).__init__()
        self.num_actions = num_actions
        self.stoch_size = stoch_size
        self.discrete = discrete
        self.hidden_size = hidden_size
        self.unroll = unroll
        self.learn_initial = learn_initial
        self.uniform_mix = uniform_mix
        self.action_clip = action_clip

        # Image Step In: [stoch * discrete, action] -> hidden
        stoch_flat_size = self.stoch_size * self.discrete if self.discrete else self.stoch_size
        self.img_in = modules.Linear(
            in_features=stoch_flat_size + self.num_actions,
            out_features=self.hidden_size,
            weight_init=weight_init,
            bias_init=bias_init
        )

        # GRU Cell: [img_in_hidden, deter] -> [reset, cand, update]
        self.gru = modules.Linear(
            in_features=self.hidden_size + self.hidden_size,
            out_features=3 * self.hidden_size,
            weight_init=weight_init,
            bias_init=bias_init
        )

        # Image Step Out: deter -> logits
        self.img_out = modules.Linear(
            in_features=self.hidden_size,
            out_features=self.stoch_size * self.discrete if self.discrete else 2 * self.stoch_size,
            weight_init=dist_weight_init,
            bias_init=dist_bias_init
        )

        # Obs Step Out: [deter, embed] -> logits
        # Assuming embed size is same as concat output of encoder.
        # However, to be generic, we can just use a large hidden size and MLP or assume embed size.
        # DreamerV3 dynamically sizes `obs_out` based on embed input. PyTorch requires fixed size.
        # We will use LazyLinear so it automatically figures out embed_size on first forward!
        self.obs_out = nn.LazyLinear(
            out_features=self.stoch_size * self.discrete if self.discrete else 2 * self.stoch_size,
        )

        if self.learn_initial:
            self.initial_deter_param = nn.Parameter(torch.zeros(self.hidden_size))
            stoch_dim = self.stoch_size * self.discrete if self.discrete else self.stoch_size
            self.initial_stoch_param = nn.Parameter(torch.zeros(stoch_dim))

    def _gru_cell(self, x, deter):
        # x: input from img_in (B, D) or (B, 1, D)
        # deter: previous deterministic state (B, 1, D) or (B, D)
        if x.dim() != deter.dim():
            if x.dim() < deter.dim():
                x = x.unsqueeze(1)
            else:
                deter = deter.unsqueeze(1)
        gru_input = torch.cat([deter, x], dim=-1)
        gates = self.gru(gru_input)
        reset, cand, update = torch.chunk(gates, 3, dim=-1)

        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        # DreamerV3 update gate is sigmoid(update - 1)
        update = torch.sigmoid(update - 1.0)

        deter_new = update * cand + (1 - update) * deter
        return deter_new

    def _stats(self, x):
        if self.discrete:
            logit = x.view(x.shape[:-1] + (self.stoch_size, self.discrete))
            return {"logits": logit}
        else:
            mean, std = torch.chunk(x, 2, dim=-1)
            std = 2 * torch.sigmoid(std / 2) + 0.1
            return {"mean": mean, "std": std}

    def get_dist(self, stats):
        if self.discrete:
            return torch.distributions.Independent(
                distributions.OneHotDist(logits=stats['logits'], uniform_mix=self.uniform_mix), 1
            )
        else:
            return torch.distributions.Independent(
                torch.distributions.Normal(stats['mean'], stats['std']), 1
            )

    def get_stoch(self, deter):
        x = self.img_out(deter)
        stats = self._stats(x)
        dist = self.get_dist(stats)
        stoch = dist.mode()
        return stoch

    def initial(self, batch_size=1, seq_length=1, dtype=torch.float32, device="cpu", detach_learned=False):
        initial_state = AttrDict(
            deter=torch.zeros(batch_size, seq_length, self.hidden_size, dtype=dtype, device=device),
            hidden=None
        )
        if self.discrete:
            initial_state.logits = torch.zeros(batch_size, seq_length, self.stoch_size, self.discrete, dtype=dtype, device=device)
            initial_state.stoch = torch.zeros(batch_size, seq_length, self.stoch_size, self.discrete, dtype=dtype, device=device)
        else:
            initial_state.mean = torch.zeros(batch_size, seq_length, self.stoch_size, dtype=dtype, device=device)
            initial_state.std = torch.ones(batch_size, seq_length, self.stoch_size, dtype=dtype, device=device)
            initial_state.stoch = torch.zeros(batch_size, seq_length, self.stoch_size, dtype=dtype, device=device)

        if self.learn_initial:
            deter_param = self.initial_deter_param.to(device)
            stoch_param = self.initial_stoch_param.to(device)
            
            initial_state.deter = deter_param.reshape(1, 1, -1).repeat(batch_size, seq_length, 1)
            if self.discrete:
                initial_state.logits = stoch_param.reshape(1, 1, self.stoch_size, self.discrete).repeat(batch_size, seq_length, 1, 1)
                initial_state.stoch = self.get_dist(AttrDict(logits=initial_state.logits)).sample()
            else:
                initial_state.mean = stoch_param.reshape(1, 1, -1).repeat(batch_size, seq_length, 1)
                initial_state.std = torch.ones_like(initial_state.mean)
                initial_state.stoch = self.get_dist(AttrDict(mean=initial_state.mean, std=initial_state.std)).sample()
            if detach_learned:
                initial_state.deter = initial_state.deter.detach()
                initial_state.stoch = initial_state.stoch.detach()
        return initial_state
    def img_step(self, prev_state, prev_action):
        # Enforce 2D for single step
        if prev_state is not None:
            prev_state = {k: v.squeeze(1) if isinstance(v, torch.Tensor) and v.dim() >= 3 and v.shape[1] == 1 else v for k, v in prev_state.items()}
        if isinstance(prev_action, torch.Tensor) and prev_action.dim() >= 3 and prev_action.shape[1] == 1:
            prev_action = prev_action.squeeze(1)
        prev_action = self._match_action_dim(prev_action)

        prev_stoch = prev_state["stoch"]
        if isinstance(self.action_clip, (int, float)) and self.action_clip > 0.0 and prev_action is not None:
            clip_val = torch.clip(torch.abs(prev_action), min=self.action_clip)
            prev_action = prev_action * (self.action_clip / clip_val).detach()

        if self.discrete:
            # Flatten stoch: (B, stoch, disc) -> (B, stoch * disc)
            if prev_stoch.shape[-2:] == (self.stoch_size, self.discrete):
                prev_stoch = prev_stoch.reshape(prev_stoch.shape[:-2] + (-1,))

        # Robust dimension matching for concatenation
        if prev_action is not None and prev_stoch.dim() != prev_action.dim():
            if prev_stoch.dim() == 2 and prev_action.dim() == 1:
                prev_action = prev_action.unsqueeze(0)
            elif prev_stoch.dim() == 3 and prev_action.dim() == 2:
                prev_action = prev_action.unsqueeze(1)

        x = torch.cat([prev_stoch, prev_action], dim=-1) if prev_action is not None else prev_stoch
        x = self.img_in(x)
        deter = self._gru_cell(x, prev_state["deter"])

        x_out = self.img_out(deter)
        stats = self._stats(x_out)
        dist = self.get_dist(stats)
        stoch = dist.rsample()

        prior = AttrDict({"stoch": stoch, "deter": deter, **stats, "hidden": None})
        return prior

    def obs_step(self, prev_state, prev_action, embed, is_first):
        # Enforce 2D for single step
        if prev_state is not None:
            prev_state = {k: v.squeeze(1) if isinstance(v, torch.Tensor) and v.dim() >= 3 and v.shape[1] == 1 else v for k, v in prev_state.items()}
        if isinstance(prev_action, torch.Tensor) and prev_action.dim() >= 3 and prev_action.shape[1] == 1:
            prev_action = prev_action.squeeze(1)
        prev_action = self._match_action_dim(prev_action)
        if isinstance(embed, torch.Tensor) and embed.dim() >= 3 and embed.shape[1] == 1:
            embed = embed.squeeze(1)
            
        if is_first is not None:
            is_first = is_first.to(prev_action.dtype) if prev_action is not None else is_first.float()

        if isinstance(self.action_clip, (int, float)) and self.action_clip > 0.0 and prev_action is not None:
            clip_val = torch.clip(torch.abs(prev_action), min=self.action_clip)
            prev_action = prev_action * (self.action_clip / clip_val).detach()

        # We need a clean initial state for reset
        init_state = self.initial(batch_size=prev_action.shape[0], seq_length=1, dtype=prev_action.dtype, device=prev_action.device)
        # Squeeze seq_length
        init_state = {k: v.squeeze(1) if v is not None else None for k, v in init_state.items()}

        mask = 1.0 - is_first
        masked_prev_state = {}
        for k, v in prev_state.items():
            if v is not None:
                mask_v = mask.view(*mask.shape, *([1] * (v.dim() - mask.dim())))
                if k in init_state:
                    masked_prev_state[k] = v * mask_v + init_state[k] * (1.0 - mask_v)
                else:
                    # Handle extra keys (like prev_action) added by agent wrapper
                    masked_prev_state[k] = v * mask_v
            else:
                masked_prev_state[k] = None

        mask_a = mask.view(*mask.shape, *([1] * (prev_action.dim() - mask.dim())))
        masked_prev_action = prev_action * mask_a

        prior = self.img_step(masked_prev_state, masked_prev_action)
        
        # Align embed dimension with prior['deter']
        if embed.dim() != prior["deter"].dim():
            if embed.dim() < prior["deter"].dim():
                embed = embed.unsqueeze(1)
            else:
                # This case is unlikely given RSSM structure, but handle for robustness
                prior["deter"] = prior["deter"].unsqueeze(1)

        x = torch.cat([prior["deter"], embed], dim=-1)
        x_out = self.obs_out(x)
        stats = self._stats(x_out)
        dist = self.get_dist(stats)
        stoch = dist.rsample()

        post = AttrDict({"stoch": stoch, "deter": prior["deter"], **stats, "hidden": None})
        return post, prior

    def observe(self, states, prev_actions, is_firsts, prev_state=None, is_firsts_hidden=None, return_blocks_deter=False):
        """
        states: dict containing 'stoch' and 'logits' for encoder embed (we will use stoch flat as embed for simplicity)
        """
        # RSSM Observe expects a loop over time
        batch_size = prev_actions.shape[0]
        seq_len = prev_actions.shape[1]
        prev_actions = self._match_action_dim(prev_actions)

        if prev_state is None:
            prev_state = self.initial(batch_size=batch_size, seq_length=1, dtype=prev_actions.dtype, device=prev_actions.device)
            prev_state = {k: v.squeeze(1) if v is not None else None for k, v in prev_state.items()}

        keys = ["stoch", "deter", "logits"] if self.discrete else ["stoch", "deter", "mean", "std"]
        posts = {k: [] for k in keys}
        priors = {k: [] for k in keys}

        # Prepare embeds
        if "latent" in states:
            embeds = states["latent"]
        else:
            embeds = states["stoch"].flatten(-2, -1) if states["stoch"].dim() == 4 else states["stoch"]

        state = prev_state
        for t in range(seq_len):
            action_t = prev_actions[:, t]
            embed_t = embeds[:, t]
            is_first_t = is_firsts[:, t]

            post, prior = self.obs_step(state, action_t, embed_t, is_first_t)

            for k in posts.keys():
                posts[k].append(post[k])
                priors[k].append(prior[k])

            state = post

        posts = {k: torch.stack(v, dim=1) for k, v in posts.items()}
        priors = {k: torch.stack(v, dim=1) for k, v in priors.items()}
        posts["hidden"] = None
        priors["hidden"] = None

        return posts, priors

    def _match_action_dim(self, actions):
        if actions is None or actions.shape[-1] == self.num_actions:
            return actions
        if actions.shape[-1] > self.num_actions:
            return actions[..., :self.num_actions]
        pad = self.num_actions - actions.shape[-1]
        return F.pad(actions, (0, pad))

    def imagine(self, p_net, prev_state, img_steps=1, is_firsts=None, is_firsts_hidden=None, actions=None):
        state = {k: v.squeeze(1) if isinstance(v, torch.Tensor) and v.dim() > 2 else v for k, v in prev_state.items()}

        policy = lambda s: p_net(self.get_feat(s).detach()).rsample()
        if actions is None:
            state["action"] = policy(state)
        else:
            state["action"] = actions[:, 0]

        img_states = {k: [v] for k, v in state.items() if k != "hidden" and v is not None}

        for h in range(img_steps):
            prior = self.img_step(state, state["action"])
            if actions is None or h == img_steps - 1:
                prior["action"] = policy(prior)
            else:
                prior["action"] = actions[:, h + 1]

            state = prior
            for k, v in state.items():
                if k != "hidden" and v is not None:
                    img_states[k].append(v)

        img_states = {k: torch.stack(v, dim=1) for k, v in img_states.items()}
        img_states["hidden"] = None
        return img_states

    def get_feat(self, state, blocks_deter_id=None):
        stoch = state["stoch"]
        deter = state["deter"]

        # Flatten discrete stochastic representations
        if self.discrete:
            # Handle (B, L, stoch, disc), (B, stoch, disc), or even with extra singleton dims
            if stoch.shape[-2:] == (self.stoch_size, self.discrete):
                stoch = stoch.reshape(stoch.shape[:-2] + (-1,))

        # Squeeze singleton temporal dimensions if they leaked in
        # We want (B, L, D) or (B, D)
        if stoch.dim() > 3 and stoch.shape[2] == 1:
            stoch = stoch.squeeze(2)
        if deter.dim() > 3 and deter.shape[2] == 1:
            deter = deter.squeeze(2)

        # Broadcast missing time dimensions if necessary
        if stoch.dim() != deter.dim():
            if stoch.dim() == 3 and deter.dim() == 2:
                deter = deter.unsqueeze(1).expand(-1, stoch.shape[1], -1)
            elif stoch.dim() == 2 and deter.dim() == 3:
                stoch = stoch.unsqueeze(1).expand(-1, deter.shape[1], -1)

        return torch.cat([stoch, deter], dim=-1)

    def slice_hidden(self, hidden):
        return None
