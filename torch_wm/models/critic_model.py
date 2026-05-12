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

"""Critic Model — Value function estimation.

Single Responsibility: Compute value prediction loss against TD(λ) returns,
weighted by trajectory survival probability and regularized against slow target.

Aligned with DreamerV3 JAX VFunction.loss():
  loss  = -dist.log_prob(sg(target))
  reg   = -dist.log_prob(sg(slow(traj).mean()))
  loss += slowreg_scale * reg
  loss  = (loss * sg(traj["weight"])).mean()
  loss *= loss_scales.critic

Extracted from twister.py inner class for SOLID compliance.
"""

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
        feats = self.outer.detached_feats[:, :-1]
        
        # 1. Value prediction loss (log-prob of returns under value distribution)
        values_dist = self.outer.value_network(feats)
        loss = -values_dist.log_prob(self.outer.detached_returns)

        # 2. Slow regularization (DreamerV3 parity)
        # Anchors critic to slowly-updated target to prevent catastrophic forgetting
        slow_dist = self.outer.v_target(feats)
        slow_mean = slow_dist.mean() if callable(slow_dist.mean) else slow_dist.mean
        reg = -values_dist.log_prob(slow_mean.detach())
        slowreg_scale = self.outer.config.get("loss_scales", {}).get("slowreg", 1.0)
        loss = loss + slowreg_scale * reg

        # 3. Weight by trajectory survival probability (set by ActorModel)
        weights = getattr(self.outer, "detached_weights", None)
        if weights is not None:
            loss = (loss.squeeze(-1) * weights).mean()
        else:
            loss = loss.mean()

        # 4. Scale by critic loss scale
        critic_scale = self.outer.config.get("loss_scales", {}).get("critic", 1.0)
        critic_loss = loss * critic_scale

        self.add_loss("critic", critic_loss)
        return critic_loss
