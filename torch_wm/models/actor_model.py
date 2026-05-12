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

"""Actor Model — Policy optimization via REINFORCE + entropy regularization.

Single Responsibility: Imagine future states via TSSM, compute TD(λ) returns,
optimize policy network using advantage-weighted REINFORCE.

Extracted from twister.py inner class for SOLID compliance.
"""

import torch
from torch_wm.core.base_model import Model as BaseModel
from carla_env.toolkit.utils import get_logger

log = get_logger(log_dir=".", job_name="actor_model")


class ActorModel(BaseModel):
    """Imagines future trajectories and optimizes the policy network.

    Args:
        outer: Reference to the parent WMAgent model for shared networks.
    """

    def __init__(self, outer):
        super().__init__(name="Actor Model")
        object.__setattr__(self, "outer", outer)

    def forward(self, inputs):
        # Handle dict or tuple inputs
        if isinstance(inputs, dict):
            s = {k: v for k, v in inputs.items() if k not in ["is_first", "is_last", "is_terminal", "action", "reward", "discount"]}
            a = inputs.get("action")
            r = inputs.get("reward")
            d = 1.0 - inputs.get("discount", torch.ones_like(r))
            f = inputs.get("is_first")
        else:
            s, a, r, d, f = inputs

        img_states = self.outer.dynamics_model.imagine(self.outer.policy_network, self.outer.detached_posts, self.outer.config.get("H", 15), self.outer.detached_is_firsts, self.outer.detached_is_firsts_hidden)
        feats = self.outer.dynamics_model.get_feat(img_states)

        # 1. Predict rewards, values and discounts from World Model heads
        # We use expectation instead of mode to provide a continuous gradient.
        # Handle both callable .mean() and property .mean for different dist types.
        model_rewards_dist = self.outer.reward_network(feats)
        model_rewards_mean = model_rewards_dist.mean() if callable(model_rewards_dist.mean) else model_rewards_dist.mean

        if self.outer.config.get("target_value_reg", False):
            values_dist = self.outer.value_network(feats)
        else:
            values_dist = self.outer.v_target(feats)
        values_mean = values_dist.mean() if callable(values_dist.mean) else values_dist.mean

        discounts_dist = self.outer.continue_network(feats)
        discounts_mean = discounts_dist.mean() if callable(discounts_dist.mean) else discounts_dist.mean

        # 2. Prepare discounts (B', 1+H, 1)
        # d is already the CONTINUATION flag from agent.py: d = 1.0 - is_terminal
        # d=1 means alive, d=0 means dead/terminal.
        d_flat = d.reshape(-1).float()
        continuation = d_flat.unsqueeze(-1).unsqueeze(-1).detach() # (B', 1, 1)

        # Combine true continuation at t=0 with predicted discounts for t=1..H
        full_discounts = torch.cat([continuation, discounts_mean[:, 1:]], dim=1)

        # 3. Compute lambda returns (B', H, 1)
        # DreamerV3 parity: use horizon-derived discount, NOT gamma
        H = self.outer.config.get("H", 15)
        discount = 1.0 - 1.0 / H
        returns = self.outer.compute_td_lambda(
            rewards=model_rewards_mean[:, 1:],
            values=values_mean[:, 1:],
            discounts=discount * full_discounts[:, 1:]
        )

        # 4. Compute Advantage and Normalize
        offset, invscale = self.outer.update_perc(returns)
        normed_returns = (returns - offset) / invscale
        normed_base = (values_mean[:, :-1] - offset) / invscale
        advantage = (normed_returns - normed_base).detach().squeeze(-1) # (B', H)

        # 5. Policy Optimization
        policy_dist = self.outer.policy_network(feats.detach())
        policy_log_prob = policy_dist.log_prob(img_states["action"].detach())[:, :-1]

        # Compute trajectory weights (B', H)
        # DreamerV3 parity: weight = cumprod(discount * cont, dim=time) / discount
        weights = torch.cumprod(discount * full_discounts, dim=1) / discount
        weights = weights[:, :-1].squeeze(-1).detach()

        # Store weights for CriticModel (critic needs trajectory weighting too)
        self.outer.detached_weights = weights

        reinforce_loss = - (weights * policy_log_prob * advantage).mean()

        # 6. Entropy Regularization
        entropy = policy_dist.entropy()[:, :-1]
        
        # DreamerV3 parity: use 'actent' key (JAX uses config.actent = 1e-2)
        entropy_scale = self.outer.config.get("run", {}).get("actent")
        if entropy_scale is None:
            entropy_scale = self.outer.config.get("actent", 1e-2)
        entropy_scale = float(entropy_scale)
        
        # One-time debug to verify config
        if not hasattr(self, '_ent_debug_done'):
            self._ent_debug_done = True
            log.info(f"[ACTOR DEBUG] actent={entropy_scale}, raw_entropy_mean={entropy.mean().item():.6f}, max_entropy={3.784:.3f}")
        
        # Periodic actor signal debug (every 50 train steps)
        if not hasattr(self, '_act_log_count'):
            self._act_log_count = 0
        self._act_log_count += 1
        if self._act_log_count % 50 == 0:
            with torch.no_grad():
                raw_ret_mean = returns.mean().item()
                raw_ret_std = returns.std().item()
                raw_val_mean = values_mean[:, :-1].mean().item()
                raw_val_std = values_mean[:, :-1].std().item()
                raw_rew_mean = model_rewards_mean[:, 1:].mean().item()
            log.info(f"[ACTOR SIGNAL] step={self._act_log_count} | adv_mean={advantage.mean().item():.4f} adv_std={advantage.std().item():.4f} | entropy={entropy.mean().item():.3f} | reinforce={reinforce_loss.item():.4f}")
            log.info(f"  -> returns: mean={raw_ret_mean:.3f} std={raw_ret_std:.3f} | values: mean={raw_val_mean:.3f} std={raw_val_std:.3f} | rew_mean={raw_rew_mean:.3f}")
            
        entropy_loss = - entropy_scale * (weights * entropy).mean()

        # 7. Action Smoothness Penalty (DreamerV3 parity)
        # Penalizes jerky actions in imagination to encourage smooth trajectories
        smoothness_scale = self.outer.config.get("run", {}).get("imag_smoothness_scale", 1e-4)
        actions = img_states["action"].detach()
        action_diffs = actions[:, 1:] - actions[:, :-1]
        smoothness_penalty = (weights * action_diffs.pow(2).mean(-1)).mean()
        smoothness_loss = smoothness_scale * smoothness_penalty
        
        actor_loss = reinforce_loss + entropy_loss + smoothness_loss
        
        self.add_loss("actor", actor_loss)
        self.add_loss("actor_reinforce", reinforce_loss)
        self.add_loss("actor_entropy", entropy_loss)
        self.add_loss("actor_smoothness", smoothness_loss)
        
        self.outer.detached_feats = feats.detach()
        self.outer.detached_returns = returns.detach()
        return actor_loss
