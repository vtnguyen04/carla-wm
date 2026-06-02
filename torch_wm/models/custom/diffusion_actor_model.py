import torch
import torch.nn.functional as F

from torch_wm.core.base_model import Model as BaseModel
from torch_wm.core.registry import ModuleRegistry


@ModuleRegistry.register('diffusion_actor')
class DiffusionActorModel(BaseModel):
    """SiT (Flow-Matching) Actor trained via REINFORCE with saved ODE logits.

    Aligned 1:1 with TWISTER/DreamerV3 actor training (twister.py lines 994-1076),
    with one adaptation: instead of re-evaluating the policy (which gives inconsistent
    logits due to stochastic ODE noise), we use the SAVED logits from the same ODE
    run that produced the actions during imagination.

    Gradient path: actor_loss → log_prob → saved_logits → ODE → SiT MLP params
    (bypasses dynamics entirely — no vanishing gradient through TSSM transformer)
    """

    def __init__(self, outer):
        super().__init__(name="Diffusion Actor Model")
        object.__setattr__(self, "outer", outer)

    def forward(self, inputs):
        # Handle dict or tuple inputs
        if isinstance(inputs, dict):
            d = 1.0 - inputs.get("discount", torch.ones_like(inputs.get("reward")))
        else:
            _, _, _, d, _ = inputs

        # --- OOM Prevention: Subsample batch for SiT ODE ---
        d_flat = d.reshape(-1).float()
        B_prime = d_flat.shape[0]
        max_imag_batch = self.outer.config.get("max_imagination_batch", 128)

        if B_prime > max_imag_batch:
            idx = torch.randperm(B_prime, device=self.device)[:max_imag_batch]
            detached_posts = {}
            for k, v in self.outer.detached_posts.items():
                if isinstance(v, torch.Tensor):
                    detached_posts[k] = v[idx]
                elif k == "hidden" and v is not None:
                    detached_posts[k] = [(kt[idx], vt[idx]) for kt, vt in v]
                else:
                    detached_posts[k] = None
            detached_is_firsts = self.outer.detached_is_firsts[idx]
            detached_is_firsts_hidden = (
                self.outer.detached_is_firsts_hidden[idx]
                if isinstance(self.outer.detached_is_firsts_hidden, torch.Tensor)
                else None
            )
            d_flat = d_flat[idx]
        else:
            detached_posts = self.outer.detached_posts
            detached_is_firsts = self.outer.detached_is_firsts
            detached_is_firsts_hidden = self.outer.detached_is_firsts_hidden

        #######################################################################
        # 1. Imagine H next states (TWISTER line 1001-1007)
        # imagine() now also saves "action_logits" — the raw ODE output
        # before discretization, for consistent REINFORCE log_prob.
        #######################################################################
        img_states = self.outer.dynamics_model.imagine(
            self.outer.policy_network,
            detached_posts,
            self.outer.config.get("H", 15),
            detached_is_firsts,
            detached_is_firsts_hidden,
        )
        # Get feat (B', 1+H, Dfeat) — TWISTER line 1010
        feats = self.outer.dynamics_model.get_feat(img_states)

        #######################################################################
        # 2. Predict rewards, values, discounts (TWISTER lines 1013-1027)
        #######################################################################
        model_rewards = self.outer.reward_network(feats)
        model_rewards_mode = model_rewards.mode() if callable(model_rewards.mode) else model_rewards.mode

        if self.outer.config.get("target_value_reg", False):
            values = self.outer.value_network(feats)
        else:
            values = self.outer.v_target(feats)
        values_mode = values.mode() if callable(values.mode) else values.mode

        discounts = self.outer.continue_network(feats)
        discounts_mode = discounts.mode() if callable(discounts.mode) else discounts.mode

        # Override first discount with true terminal (TWISTER line 1026-1027)
        true_first = (1.0 - d_flat).unsqueeze(-1).unsqueeze(-1).detach()
        discounts_full = torch.cat([true_first, discounts_mode[:, 1:]], dim=1)

        #######################################################################
        # 3. Trajectory weights (TWISTER line 1034)
        #######################################################################
        gamma = self.outer.config.get("gamma", 0.997)
        weights = torch.cumprod(gamma * discounts_full, dim=1).detach() / gamma

        #######################################################################
        # 4. Lambda returns + Advantage (TWISTER lines 1037-1048)
        #######################################################################
        returns = self.outer.compute_td_lambda(
            rewards=model_rewards_mode[:, 1:],
            values=values_mode[:, 1:],
            discounts=gamma * discounts_full[:, 1:],
        )

        # Update Perc (TWISTER line 1041)
        if hasattr(self.outer, "update_perc"):
            offset, invscale = self.outer.update_perc(returns)
        else:
            offset = returns.mean().detach()
            invscale = returns.std().detach() + 1e-8

        # Norm Returns (TWISTER line 1044-1045)
        normed_returns = (returns - offset) / invscale
        normed_base = (values_mode[:, :-1] - offset) / invscale

        # Advantage (B', H) — TWISTER line 1048
        advantage = (normed_returns - normed_base).squeeze(dim=-1)

        #######################################################################
        # 5. REINFORCE Actor Loss (TWISTER lines 1051-1070)
        #
        # TWISTER uses:
        #   policy_dist = self.policy_network(feats.detach())   ← re-evaluate
        #
        # For SiT (flow-matching), re-evaluate gives DIFFERENT logits each time
        # (stochastic ODE noise). So instead, we use the SAVED logits from the
        # same ODE run that produced the actions. This is equivalent to TWISTER's
        # re-evaluation because feats.detach() ensures no gradient through dynamics
        # in both cases.
        #######################################################################
        from torch_wm.modules.networks.sit_policy import SiTDistWrapper

        # Policy Dist from saved logits (B', 1+H, A) — replaces TWISTER line 1051
        saved_logits = img_states["action_logits"]
        policy_dist = SiTDistWrapper(saved_logits, discrete=self.outer.policy_network.discrete)

        if self.outer.policy_network.discrete:
            actor_loss = (
                policy_dist.log_prob(img_states["action"].detach())[:, :-1]
                * advantage.detach()
            )
        else:
            # DPG: gradients flow through dynamics and value model into the policy
            actor_loss = advantage

        # Entropy bonus — TWISTER lines 1062-1064
        # Config lookup: eta_entropy > actent > default 3e-4
        eta_entropy = float(self.outer.config.get("eta_entropy",
                            self.outer.config.get("actent", 3e-4)))
        policy_ent = policy_dist.entropy()[:, :-1]
        actor_loss = actor_loss + eta_entropy * policy_ent

        # Apply weights — TWISTER line 1067
        actor_loss = actor_loss * weights[:, :-1].squeeze(dim=-1)

        # Negate for minimization — TWISTER line 1070
        total_loss = -actor_loss.mean()

        #######################################################################
        # 6. Metrics + shared state for critic
        #######################################################################
        with torch.no_grad():
            entropy_val = policy_ent.mean()
            max_entropy = torch.log(torch.tensor(float(saved_logits.shape[-1]),
                                                  device=saved_logits.device))
            entropy_ratio = entropy_val / max_entropy  # 1.0 = uniform, 0.0 = collapsed

            # Action diversity: count unique actions in batch
            actions_flat = img_states["action"][:, :-1].argmax(dim=-1).reshape(-1)
            n_unique = actions_flat.unique().numel()
            n_possible = saved_logits.shape[-1]

        self.add_metric("actor/advantage_mean", advantage.mean().item())
        self.add_metric("actor/advantage_std", advantage.std().item())
        self.add_metric("actor/entropy", entropy_val.item())
        self.add_metric("actor/entropy_ratio", entropy_ratio.item())
        self.add_metric("actor/action_diversity", n_unique / n_possible)
        self.add_metric("actor/returns_mean", returns.mean().item())
        self.add_metric("actor/eta_entropy", eta_entropy)

        self.add_loss("actor", total_loss)

        # Shared state consumed by CriticModel
        self.outer.detached_feats = feats.detach()
        self.outer.detached_returns = returns.detach()
        self.outer.detached_weights = weights.detach()

        return total_loss
