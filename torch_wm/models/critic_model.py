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

Single Responsibility: Compute value prediction loss against TD(λ) returns.

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
        values = self.outer.value_network(self.outer.detached_feats[:, :-1])
        critic_loss = - values.log_prob(self.outer.detached_returns).mean()
        self.add_loss("critic", critic_loss)
        return critic_loss
