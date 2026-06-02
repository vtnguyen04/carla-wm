# Copyright 2025, Maxime Burchi.
# Modifications copyright 2026, Vo Thanh Nguyen.
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
        # TWISTER uses .mode() for all heads — NOT .mean().
        # For Bernoulli (continue_network), .mode returns 0 or 1 (alive/dead),
        # while .mean() returns a continuous probability that dilutes discounts.
        model_rewards_dist = self.outer.reward_network(feats)
        model_rewards_mode = model_rewards_dist.mode() if callable(model_rewards_dist.mode) else model_rewards_dist.mode

        if self.outer.config.get("target_value_reg", False):
            values_dist = self.outer.value_network(feats)
        else:
            values_dist = self.outer.v_target(feats)
        values_mode = values_dist.mode() if callable(values_dist.mode) else values_dist.mode

        # CRITICAL: .mode (property, not callable) for Bernoulli — returns 0/1
        discounts_dist = self.outer.continue_network(feats)
        discounts_mode = discounts_dist.mode if not callable(discounts_dist.mode) else discounts_dist.mode()

        # 2. Prepare discounts (B', 1+H, 1)
        # d is dones (is_terminal): 0=alive, 1=terminal
        # continuation = 1 - dones: probability of still being alive
        d_flat = d.reshape(-1).float()
        continuation = (1.0 - d_flat).unsqueeze(-1).unsqueeze(-1).detach() # (B', 1, 1)

        # Combine true continuation at t=0 with predicted discounts for t=1..H
        full_discounts = torch.cat([continuation, discounts_mode[:, 1:]], dim=1)

        # 3. Compute lambda returns (B', H, 1)
        # TWISTER uses config.gamma directly (not DreamerV3's horizon-based calc)
        gamma = self.outer.config.get("gamma", 0.997)
        
        returns = self.outer.compute_td_lambda(
            rewards=model_rewards_mode[:, 1:],
            values=values_mode[:, 1:],
            discounts=gamma * full_discounts[:, 1:]
        )

        # 4. Compute Advantage and Normalize
        offset, invscale = self.outer.update_perc(returns)
        normed_returns = (returns - offset) / invscale
        normed_base = (values_mode[:, :-1] - offset) / invscale
        advantage = (normed_returns - normed_base).detach().squeeze(-1) # (B', H)

        # 5. Policy Optimization
        if self.outer.config.get("discrete_actions", True):
            actor_grad = self.outer.config.get("run", {}).get("actor_grad_disc", "reinforce")
        else:
            actor_grad = self.outer.config.get("run", {}).get("actor_grad_cont", "backprop")
            
        policy_dist = self.outer.policy_network(feats.detach())
        
        # Compute trajectory weights (B', H)
        # DreamerV3 parity: weight = cumprod(discount * cont, dim=time) / discount
        weights = torch.cumprod(gamma * full_discounts, dim=1).detach() / gamma
        weights = weights[:, :-1].squeeze(-1).detach()

        # Store weights for CriticModel (critic needs trajectory weighting too)
        self.outer.detached_weights = weights
        
        if actor_grad in ["backprop", "dynamics"]:
            # Dynamics Backprop: Gradient flows directly through the value and transition models
            policy_loss = - (weights * advantage).mean()
        elif actor_grad == "reinforce":
            # REINFORCE: Gradient flows through log probabilities
            policy_log_prob = policy_dist.log_prob(img_states["action"].detach())[:, :-1]
            policy_loss = - (weights * policy_log_prob * advantage.detach()).mean()
        else:
            raise NotImplementedError(f"Unknown actor_grad: {actor_grad}")

        # 6. Entropy Regularization
        entropy = policy_dist.entropy()[:, :-1]
        
        # DreamerV3 parity: use 'actent' key (JAX uses config.actent = 3e-4)
        entropy_scale = self.outer.config.get("run", {}).get("actent")
        if entropy_scale is None:
            entropy_scale = self.outer.config.get("actent", 3e-4)
        entropy_scale = float(entropy_scale)
        
        # Structured metrics (routed to W&B/TB via agent.py)
        self.add_metric("actor/advantage_mean", advantage.mean().item())
        self.add_metric("actor/advantage_std", advantage.std().item())
        self.add_metric("actor/entropy_mean", entropy.mean().item())
        self.add_metric("actor/entropy_scale", entropy_scale)
        if 'policy_log_prob' in locals():
            self.add_metric("actor/policy_log_prob", policy_log_prob.mean().item())
        
        entropy_loss = - entropy_scale * (weights * entropy).mean()

        # 7. Action Smoothness Penalty (DreamerV3 parity)
        # Penalizes jerky actions in imagination to encourage smooth trajectories.
        # In discrete spaces, this acts as a 'stay-the-same-action' regularizer.
        smoothness_scale = self.outer.config.get("run", {}).get("imag_smoothness_scale")
        if smoothness_scale is None:
            smoothness_scale = self.outer.config.get("imag_smoothness_scale", 1e-4)
        smoothness_scale = float(smoothness_scale)

        actions = img_states["action"].detach()
        # Calculate diffs between consecutive actions in the imagined trajectory
        action_diffs = actions[:, 1:] - actions[:, :-1]
        
        # Smoothness penalty is the mean squared difference, weighted by trajectory importance
        smoothness_penalty = (weights * action_diffs.pow(2).mean(-1)).mean()
        smoothness_loss = smoothness_scale * smoothness_penalty
        
        actor_loss = policy_loss + entropy_loss + smoothness_loss
        
        self.add_loss("actor", actor_loss)
        self.add_loss("actor_policy", policy_loss)
        self.add_loss("actor_entropy", entropy_loss)
        self.add_loss("actor_smoothness", smoothness_loss)
        
        self.outer.detached_feats = feats.detach()
        self.outer.detached_returns = returns.detach()
        return actor_loss
