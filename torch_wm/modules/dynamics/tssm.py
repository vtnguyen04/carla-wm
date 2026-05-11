from typing import List, Dict, Tuple, Optional, Any, Union
# Copyright 2025, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch
import torch
import torch.nn as nn

# NeuralNets
from torch_wm import modules
from torch_wm import distributions
from torch_wm import structs
from torch_wm.core.registry import ModuleRegistry

@ModuleRegistry.register('tssm')
class TSSM(nn.Module):

    def __init__(
            self, 
            num_actions,
            stoch_size=32, 
            act_fun=nn.SiLU,
            discrete=32, 
            learn_initial=True, 
            weight_init="dreamerv3_normal", 
            bias_init="zeros", 
            norm={"class": "LayerNorm", "params": {"eps": 1e-3}}, 
            uniform_mix=0.01, 
            action_clip=1.0, 
            dist_weight_init="xavier_uniform", 
            dist_bias_init="zeros",

            # Transformer
            hidden_size=1024,
            num_blocks=4,
            ff_ratio=4,
            num_heads=16,
            drop_rate=0.1,
            att_context_left=64,
            module_pre_norm=False,
            use_checkpointing=False,
        ):
        super(TSSM, self).__init__()

        # Params
        self.num_actions = num_actions
        self.stoch_size = stoch_size
        self.act_fun = act_fun
        self.discrete = discrete
        self.learn_initial = learn_initial
        self.weight_init_str = weight_init # Renamed to avoid conflict
        self.bias_init = bias_init
        self.norm = norm
        self.uniform_mix = uniform_mix
        self.action_clip = action_clip
        self.dist_weight_init = dist_weight_init
        self.dist_bias_init = dist_bias_init

        # Transformer
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.ff_ratio = ff_ratio
        self.num_heads = num_heads
        self.drop_rate = drop_rate
        self.att_context_left = att_context_left
        self.max_pos_encoding = 2048

        # zt + at -> Linear -> Norm -> Act -> Linear -> Norm -> et
        self.action_mixer = modules.MultiLayerPerceptron(
            dim_input=self.stoch_size * self.discrete + self.num_actions if self.discrete else self.stoch_size + self.num_actions, 
            dim_layers=[self.hidden_size, self.hidden_size],
            act_fun=[self.act_fun, None],
            weight_init=weight_init,
            bias_init=self.bias_init,
            norm=[self.norm, None] if module_pre_norm else self.norm,
            bias=self.norm is None
        )

        # Transformer et -> dt, ht
        self.transformer = modules.TransformerNetwork(
            dim_model=self.hidden_size,
            num_blocks=self.num_blocks,
            att_params={
                "class": "RelPosMultiHeadSelfAttention",
                "params": {
                    "num_heads": self.num_heads, 
                    "weight_init": "default", 
                    "bias_init": "default", 
                    "attn_drop_rate": self.drop_rate, 
                    "max_pos_encoding": self.max_pos_encoding, 
                    "causal": True
                }
            },
            emb_drop_rate=0.0,
            drop_rate=self.drop_rate,
            pos_embedding=None,
            mask=None,
            ff_ratio=self.ff_ratio,
            weight_init="default", 
            bias_init="default",
            act_fun="ReLU",
            module_pre_norm=module_pre_norm,
            use_checkpointing=use_checkpointing
        )

        # Dynamics Predictor dt -> zt
        self.dynamics_predictor = modules.Linear(
            in_features=self.hidden_size, 
            out_features=self.discrete * self.stoch_size if self.discrete else 2 * self.stoch_size,
            weight_init=self.dist_weight_init,
            bias_init=self.dist_bias_init
        )

        if self.learn_initial:
            self.initial_deter_param = nn.Parameter(torch.zeros(self.hidden_size))

    def forward_dynamics(self, prev_state, action):
        if action.dim() == 2:
            action = action.unsqueeze(1)
        hidden_len = self.get_hidden_len(prev_state.get("hidden"))
        mask = modules.return_mask(seq_len=1, hidden_len=hidden_len, left_context=self.att_context_left, right_context=0, dtype=action.dtype, device=action.device)
        return self.forward_img(prev_state, action, mask)

    def get_stoch(self, deter):
        logits = self.dynamics_predictor(deter).reshape(deter.shape[:-1] + (self.stoch_size, self.discrete))
        dist_params = {'logits': logits}
        stoch = self.get_dist(dist_params).mode()
        return stoch

    def img_step(self, prev_states, prev_actions):
        """Standard interface for single-step imagination."""
        # For TSSM, we need a mask. For 1 step, it's just the default causal mask.
        hidden_len_val = self.get_hidden_len(prev_states.get("hidden"))
        mask = modules.return_mask(
            seq_len=prev_actions.shape[1], 
            hidden_len=hidden_len_val, 
            left_context=self.att_context_left, 
            right_context=0, 
            dtype=prev_actions.dtype, 
            device=prev_actions.device
        )
        return self.forward_img(prev_states, prev_actions, mask)

    def initial(self, batch_size=1, seq_length=1, dtype=torch.float32, device="cpu", detach_learned=False):
        initial_state = structs.AttrDict(
            logits=torch.zeros(batch_size, seq_length, self.stoch_size, self.discrete, dtype=dtype, device=device),
            stoch=torch.zeros(batch_size, seq_length, self.stoch_size, self.discrete, dtype=dtype, device=device),
            deter=torch.zeros(batch_size, seq_length, self.hidden_size, dtype=dtype, device=device),
            hidden=None
        )
        if self.learn_initial:
            initial_state.deter = self.initial_deter_param.repeat(batch_size, seq_length, 1)
            initial_state.stoch = self.get_stoch(initial_state.deter) 
            if detach_learned:
                initial_state.deter = initial_state.deter.detach()
                initial_state.stoch = initial_state.stoch.detach()
        return initial_state

    def observe(self, states, prev_actions, is_firsts, prev_state=None, is_firsts_hidden=None, return_blocks_deter=False, return_att_w=False):
        prev_states = {}
        for key, value in states.items():
            if isinstance(value, torch.Tensor):
                prev_states[key] = value[:, :-1]
            elif key == "hidden" and isinstance(value, list):
                prev_states[key] = [(v[0][:, :-1], v[1][:, :-1]) if isinstance(v, tuple) else v[:, :-1] for v in value]
        if prev_state is None:
            prev_actions[:, 0] = 0.0
            prev_state = self.initial(batch_size=prev_actions.shape[0], seq_length=1, dtype=prev_actions.dtype, device=prev_actions.device)
        for key in list(prev_states.keys()):
            if key not in prev_state or prev_state[key] is None:
                if isinstance(states[key], torch.Tensor):
                    prev_state[key] = torch.zeros_like(states[key][:, :1])
                else:
                    prev_state[key] = None
            if isinstance(prev_state[key], torch.Tensor) and isinstance(prev_states[key], torch.Tensor):
                if prev_state[key].dim() != prev_states[key].dim():
                    prev_state[key] = torch.zeros_like(states[key][:, :1])
        prev_states_cat = {}
        for key, value in prev_states.items():
            if isinstance(value, torch.Tensor) and isinstance(prev_state.get(key), torch.Tensor):
                prev_states_cat[key] = torch.cat([prev_state[key], value], dim=1)
        prev_states = prev_states_cat
        prev_states["hidden"] = prev_state["hidden"]
        posts, priors = self(states, prev_states, prev_actions, is_firsts, is_firsts_hidden, return_blocks_deter=return_blocks_deter, return_att_w=return_att_w)
        return posts, priors

    def imagine(self, p_net, prev_state, img_steps=1, is_firsts=None, is_firsts_hidden=None, actions=None):
        policy = lambda s: p_net(self.get_feat(s).detach()).rsample()
        if actions is None:
            prev_state["action"] = policy(prev_state)
        else:
            prev_state["action"] = actions[:, :1]
        img_states = {"stoch": [prev_state["stoch"]], "deter": [prev_state["deter"]], "logits": [prev_state["logits"]], "action": [prev_state["action"]]}
        for h in range(img_steps):
            hidden_len_val = self.get_hidden_len(prev_state["hidden"])
            mask = modules.return_mask(seq_len=1, hidden_len=hidden_len_val, left_context=self.att_context_left, right_context=0, dtype=prev_state["action"].dtype, device=prev_state["action"].device)
            if is_firsts_hidden is not None:
                is_firts_mask = modules.return_is_firsts_mask(is_firsts=is_firsts, is_firsts_hidden=is_firsts_hidden)
                mask = mask.minimum(is_firts_mask)
                is_firsts_hidden = torch.cat([is_firsts_hidden[:, 1:], is_firsts], dim=1)
                is_firsts = torch.zeros_like(is_firsts)
            img_state = self.forward_img(prev_states=prev_state, prev_actions=prev_state["action"], mask=mask)
            if actions is None or h==img_steps-1:
                img_state["action"] = policy(img_state)
            else:
                img_state["action"] = actions[:, h+1:h+2]
            img_state["hidden"] = self.slice_hidden(img_state["hidden"])
            prev_state = img_state
            for key, value in img_state.items():
                if key != "hidden":
                    img_states[key].append(value)
        img_states = {k: torch.concat(v, dim=1) for k, v in img_states.items()}
        return img_states

    def get_feat(self, state, blocks_deter_id=None):
        stoch = state["stoch"]
        deter = state["deter"] if blocks_deter_id is None else state["blocks_deter"][blocks_deter_id]
        
        # Flatten discrete stochastic representations
        if stoch.dim() == 4:
            stoch = stoch.reshape(stoch.shape[0], stoch.shape[1], -1)
        elif stoch.dim() == 3 and self.discrete and stoch.shape[-2:] == (self.stoch_size, self.discrete):
            stoch = stoch.reshape(stoch.shape[0], -1)
            
        # Broadcast missing time dimensions if necessary
        if stoch.dim() == 3 and deter.dim() == 2:
            deter = deter.unsqueeze(1).expand(-1, stoch.shape[1], -1)
        elif stoch.dim() == 2 and deter.dim() == 3:
            stoch = stoch.unsqueeze(1).expand(-1, deter.shape[1], -1)
            
        return torch.cat([stoch, deter], dim=-1)
    
    def get_dist(self, state):
        return torch.distributions.Independent(distributions.OneHotDist(logits=state['logits'], uniform_mix=self.uniform_mix), 1)

    def slice_hidden(self, hidden):
        hidden = [(hidden_blk[0][:, -self.att_context_left:], hidden_blk[1][:, -self.att_context_left:]) for hidden_blk in hidden]
        return hidden
    
    def get_hidden_len(self, hidden):
        if hidden != None:
            return hidden[0][0].shape[1]
        else:
            return 0

    def forward_img(self, prev_states, prev_actions, mask, return_att_w=False, return_blocks_deter=False):
        if self.action_clip > 0.0:
            prev_actions = prev_actions * (self.action_clip / torch.clip(torch.abs(prev_actions), min=self.action_clip)).detach()
        batch_size = prev_actions.shape[0]
        seq_len = prev_actions.shape[1]
        stoch = prev_states["stoch"]
        # SYNC FIX: Take only the necessary history steps to match current action sequence
        stoch_flat = stoch[:, -prev_actions.shape[1]:].reshape(batch_size, prev_actions.shape[1], -1)
        prev_actions_flat = prev_actions.reshape(batch_size, prev_actions.shape[1], -1)
        
        x = self.action_mixer(torch.concat([stoch_flat, prev_actions_flat], dim=-1))
        outputs = self.transformer(x, hidden=prev_states["hidden"], mask=mask, return_hidden=True, return_att_w=return_att_w, return_blocks_x=return_blocks_deter)
        deter, hidden = outputs.x, outputs.hidden
        add_out_dict = {}
        if return_att_w:
            add_out_dict["att_w"] = outputs.att_w
        if return_blocks_deter:
            add_out_dict["blocks_deter"] = outputs.blocks_x
        logits = self.dynamics_predictor(deter).reshape(deter.shape[:-1] + (self.stoch_size, self.discrete))
        dist_params = {'logits': logits}
        stoch = self.get_dist(dist_params).rsample()
        return {"stoch": stoch, "deter": deter, "hidden": hidden, **dist_params, **add_out_dict}
    
    def forward_obs(self, deter, hidden, states):
        return {"deter": deter, "hidden": hidden, **states}

    def forward(self, states, prev_states, prev_actions, is_firsts, is_firsts_hidden=None, return_att_w=False, return_blocks_deter=False):
        if is_firsts.dim() == 3 and is_firsts.shape[-1] == 1:
            is_firsts = is_firsts.squeeze(-1)
        if prev_actions.dim() != 3:
            prev_actions = prev_actions.reshape(prev_actions.shape[0], -1, self.num_actions)
        if is_firsts.dim() == 1:
            is_firsts = is_firsts.unsqueeze(1)
        if self.action_clip > 0.0:
            prev_actions *= (self.action_clip / torch.clip(torch.abs(prev_actions), min=self.action_clip)).detach()
        mask = modules.return_mask(seq_len=prev_actions.shape[1], hidden_len=self.get_hidden_len(prev_states["hidden"]), left_context=self.att_context_left, right_context=0, dtype=prev_actions.dtype, device=prev_actions.device)
        if is_firsts.any():
            is_firsts_u = is_firsts.unsqueeze(dim=-1).to(prev_actions.dtype)
            prev_actions *= (1.0 - is_firsts_u)
            init_state = self.initial(batch_size=prev_actions.shape[0], seq_length=prev_actions.shape[1], dtype=prev_actions.dtype, device=prev_actions.device)
            for key, value in prev_states.items():
                if key == "hidden":
                    prev_states[key] = value
                else:
                    if key not in init_state:
                        init_state[key] = torch.zeros_like(value)
                    is_firsts_r = torch.reshape(is_firsts_u, is_firsts_u.shape + (1,) * (len(value.shape) - len(is_firsts_u.shape)))
                    prev_states[key] = value * (1.0 - is_firsts_r) + init_state[key] * is_firsts_r
            is_firts_mask = modules.return_is_firsts_mask(is_firsts.squeeze(dim=-1), is_firsts_hidden=is_firsts_hidden)
            mask = mask.minimum(is_firts_mask)
        prior = self.forward_img(prev_states, prev_actions, mask, return_att_w=return_att_w, return_blocks_deter=return_blocks_deter)
        post = self.forward_obs(prior["deter"], prior["hidden"], states)
        if return_att_w:
            post["att_w"] = prior["att_w"]
        if return_blocks_deter:
            post["blocks_deter"] = prior["blocks_deter"]
        return post, prior
