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
        s, a, r, d, f = inputs

        img_states = self.outer.dynamics_model.imagine(self.outer.policy_network, self.outer.detached_posts, self.outer.config.H, self.outer.detached_is_firsts, self.outer.detached_is_firsts_hidden)
        feats = self.outer.dynamics_model.get_feat(img_states)

        # Predict rewards (B', 1+H, 1)
        model_rewards = self.outer.reward_network(feats)

        # Predict Values (B', 1+H, 1)
        if self.outer.config.target_value_reg:
            values = self.outer.value_network(feats)
        else:
            values = self.outer.v_target(feats)

        # Predict Discounts (B', 1+H, 1)
        discounts_dist = self.outer.continue_network(feats)
        discounts = discounts_dist.mode() if callable(discounts_dist.mode) else discounts_dist.mode

        # Override discount prediction for the first step with the true
        # discount factor from the replay buffer.
        d_flat = d.reshape(-1).float()  # (B*L,) and cast to float for arithmetic
        true_first = (1.0 - d_flat).unsqueeze(dim=-1).unsqueeze(dim=-1)  # (B*L, 1, 1)
        discounts = torch.cat([true_first, discounts[:, 1:]], dim=1)

        # Compute lambda returns (B', H, 1)
        model_rewards_mode = model_rewards.mode() if callable(model_rewards.mode) else model_rewards.mode
        values_mode = values.mode() if callable(values.mode) else values.mode
        returns = self.outer.compute_td_lambda(rewards=model_rewards_mode[:, 1:], values=values_mode[:, 1:], discounts=self.outer.config.gamma * discounts[:, 1:])

        # Update Perc
        offset, invscale = self.outer.update_perc(returns)

        # Norm Returns using quantiles ema ~ [0:1]
        normed_returns = (returns - offset) / invscale  # 1:H+1
        normed_base = (values_mode[:, :-1] - offset) / invscale  # 0:H

        # advantage (B', H)
        advantage = (normed_returns - normed_base).squeeze(dim=-1)

        # Policy Dist (B', 1+H, A)
        policy_dist = self.outer.policy_network(feats.detach())

        # Actor Loss (REINFORCE with Entropy Regularization)
        policy_log_prob = policy_dist.log_prob(img_states["action"].detach())[:, :-1]
        reinforce_loss = - (policy_log_prob * advantage.detach()).mean()
        
        # Encourage exploration
        actor_entropy_scale = self.outer.config.get("actor_entropy", 3e-4)
        entropy_bonus = policy_dist.entropy()[:, :-1].mean()
        
        actor_loss = reinforce_loss - (actor_entropy_scale * entropy_bonus)

        self.add_loss("actor", actor_loss)
        self.outer.detached_feats = feats.detach()
        self.outer.detached_returns = returns.detach()
        return actor_loss
