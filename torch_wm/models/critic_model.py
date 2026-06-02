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

"""Critic Model — Value function estimation.

"""

import torch
from torch_wm.core.base_model import Model as BaseModel


class CriticModel(BaseModel):
    """Optimizes the value network to predict expected returns.

    Args:
        outer: Reference to the parent WMAgent model for shared networks.
    """

    def __init__(self, outer):
        super().__init__(name="Critic Model")
        object.__setattr__(self, "outer", outer)

    def forward(self, inputs):
        # Shared state from ActorModel
        feats = self.outer.detached_feats    # (B', 1+H, D)
        returns = self.outer.detached_returns  # (B', H, 1)
        weights = self.outer.detached_weights  # (B', 1+H, 1)

        # Value (B', H, 1) — TWISTER line 1105
        value_dist = self.outer.value_network(feats.detach()[:, :-1])

        # Value Loss — TWISTER line 1108
        value_loss = value_dist.log_prob(returns.detach())

        # Slow Regularization — TWISTER lines 1111-1114
        if self.outer.config.get("target_value_reg", False):
            with torch.no_grad():
                target_dist = self.outer.v_target(feats.detach()[:, :-1])
                value_target = target_dist.mode() if callable(target_dist.mode) else target_dist.mode
            slow_reg_scale = self.outer.config.get("loss_scales", {}).get(
                "slowreg", self.outer.config.get("critic_slow_reg_scale", 1.0)
            )
            value_loss = value_loss + slow_reg_scale * value_dist.log_prob(value_target.detach())

        # Weight loss — TWISTER line 1117
        # weights already has shape (B', H) from the ActorModel
        if weights.dim() == 3:
            weights = weights[:, :-1].squeeze(dim=-1)
        value_loss = value_loss * weights

        # Add Loss — TWISTER line 1120
        critic_loss = -value_loss.mean()

        # Optional critic scale
        critic_scale = self.outer.config.get("loss_scales", {}).get("critic", 1.0)
        critic_loss = critic_loss * critic_scale

        self.add_loss("critic", critic_loss)
        return critic_loss
