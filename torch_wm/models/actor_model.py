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
        # We override the first step with ground truth continuation from the buffer.
        # d is terminal flag (1=dead, 0=alive), so (1.0 - d) is continuation.
        d_flat = d.reshape(-1).float()
        true_first = (1.0 - d_flat).unsqueeze(-1).unsqueeze(-1).detach() # (B', 1, 1)

        # Combine true continuation at t=0 with predicted discounts for t=1..H
        full_discounts = torch.cat([true_first, discounts_mean[:, 1:]], dim=1)

        # 3. Compute lambda returns (B', H, 1)
        # Note: We use full_discounts[:, 1:] which corresponds to steps 1..H
        returns = self.outer.compute_td_lambda(
            rewards=model_rewards_mean[:, 1:],
            values=values_mean[:, 1:],
            discounts=self.outer.config.gamma * full_discounts[:, 1:]
        )

        # 4. Compute Advantage and Normalize
        offset, invscale = self.outer.update_perc(returns)
        normed_returns = (returns - offset) / invscale
        normed_base = (values_mean[:, :-1] - offset) / invscale
        advantage = (normed_returns - normed_base).detach().squeeze(-1) # (B', H)

        # 5. Policy Optimization
        policy_dist = self.outer.policy_network(feats.detach())
        policy_log_prob = policy_dist.log_prob(img_states["action"].detach())[:, :-1]

        reinforce_loss = - (policy_log_prob * advantage * true_first.squeeze(-1)).mean()

        # 6. Entropy Regularization
        entropy = policy_dist.entropy()[:, :-1]
        
        # Defensive configuration access for entropy parameters
        # Support both nested (config.actor.entropy) and flat (config.actor_entropy) structures
        # Also handles cases where 'run' might be missing
        entropy_scale = getattr(self.outer.config, "actor", {}).get("entropy")
        if entropy_scale is None:
            entropy_scale = getattr(self.outer.config, "run", {}).get("actor_entropy")
        if entropy_scale is None:
            entropy_scale = getattr(self.outer.config, "actor_entropy", 1e-3)
            
        entropy_loss = - entropy_scale * (entropy * true_first.squeeze(-1)).mean()
        
        actor_loss = reinforce_loss + entropy_loss
        
        self.add_loss("actor", actor_loss)
        self.add_loss("actor_reinforce", reinforce_loss)
        self.add_loss("actor_entropy", entropy_loss)
        
        self.outer.detached_feats = feats.detach()
        self.outer.detached_returns = returns.detach()
        return actor_loss
