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

"""World Model — TSSM Observation + Modular Loss Computation.

Single Responsibility: Encode observations, run TSSM observe,
compute reconstruction/reward/discount/CPC/curvature/KL losses.
Produces detached states for Actor/Critic downstream.

Extracted from twister.py inner class for SOLID compliance.
"""

import torch
from torch_wm.core.base_model import Model as BaseModel
from torch_wm.core.manager import LossManager
from torch_wm.utils.preprocessing import preprocess_obs

class WorldModel(BaseModel):
    """Encodes observations, runs TSSM observe, computes all world model losses.
    
    Args:
        outer: Reference to the parent WMAgent model for shared networks.
    """

    def __init__(self, outer):
        super().__init__(name="World Model")
        object.__setattr__(self, "outer", outer)
        self.loss_manager = LossManager(self.outer.config)

    def forward(self, inputs):
        # Handle dict inputs (standard in our agent pipeline)
        if isinstance(inputs, dict):
            s = {k: v for k, v in inputs.items() if k not in ["is_first", "is_last", "is_terminal", "action", "reward", "discount"]}
            a = inputs.get("action")
            r = inputs.get("reward")
            d = 1.0 - inputs.get("discount", torch.ones_like(r))
            f = inputs.get("is_first")
        # Handle tuple inputs (backward compat / dry-run)
        elif len(inputs) == 6:
            s, a, r, d, f, _ = inputs
        else:
            s, a, r, d, f = inputs
            
        # Preprocess observations (DRY: shared utility)
        precision = self.outer.config.get("precision", "float32")
        s = preprocess_obs(s, device=self.device, precision=precision)
                
        encoder_out = self.outer.encoder_network(s)
        latent_stoch = encoder_out["stoch"]
        latent_features = encoder_out["latent"]
        signal_features = encoder_out.get("signal", None)
        
        if f.dim() == 3 and f.shape[-1] == 1:
            f = f.squeeze(-1)
            
        posts, priors = self.outer.dynamics_model.observe(
            encoder_out, 
            a, f
        )
        feats = self.outer.dynamics_model.get_feat(posts)
        
        # TWISTER parity: decoder receives ONLY stoch (not feats=stoch+deter)
        # This forces the encoder to push all visual info into the stochastic variable,
        # preventing entropy collapse. See TWISTER line 839.
        stoch_flat = posts["stoch"].flatten(-2, -1) if posts["stoch"].dim() > 3 else posts["stoch"]
        
        if self.outer.config.dynamics_type == "rssm":
            decoder_input = torch.cat([posts["deter"], stoch_flat], dim=-1)
        else:
            decoder_input = stoch_flat
            
        model_outputs = {
            "posts": posts, "priors": priors, "feats": feats, "latent": latent_features,
            "states_rec_dist": self.outer.decoder_network(decoder_input),
            "model_rewards": self.outer.reward_network(feats),
            "model_discounts": self.outer.continue_network(feats),
            "signal_features": signal_features,
        }
        
        # Store for visualization
        self.outer._last_outputs.update(model_outputs)
        
        # Modular Loss delegation
        losses = self.loss_manager.compute_total_loss(
            model_outputs, 
            {"states": s, "actions": a, "rewards": r, "dones": d, "is_firsts": f}, 
            tssm=self.outer.dynamics_model, 
            contrastive_network=self.outer.contrastive_network, 
            encoder_network=self.outer.encoder_network, 
            ema_encoder=getattr(self.outer, 'ema_encoder', None),
            ema_dynamics=getattr(self.outer, 'ema_dynamics', None),
            config=self.outer.config
        )
        
        for name, val in losses.items():
            if name != "total": self.add_loss(name, val, 1.0)
        if "total" in losses:
            self.add_metric("total_loss", losses["total"].detach())
        
        # Collect and add metrics
        metrics = self.loss_manager.get_metrics()
        for name, val in metrics.items():
            self.add_metric(name, val)
        
        # Detach states for Actor/Critic downstream
        is_firsts_hidden_concat = f

        # K, V: (B, C+L, D) -> (B*L, C, D)
        hidden_flatten = []
        C = self.outer.config.get("att_context_left", 32)
        L = self.outer.config.get("L", self.outer.config.get("batch_length", 32))
        
        if posts.get("hidden") is not None:
            for hidden_blk in posts["hidden"]:
                k_list = []
                v_list = []
                for t in range(L):
                    end_idx = hidden_blk[0].shape[1] - L + t
                    start_idx = max(0, end_idx - C)
                    
                    k_slice = hidden_blk[0][:, start_idx:end_idx]
                    v_slice = hidden_blk[1][:, start_idx:end_idx]
                    
                    # Padding to fixed length C
                    if k_slice.shape[1] < C:
                        pad = C - k_slice.shape[1]
                        k_slice = torch.cat([k_slice.new_zeros(k_slice.shape[0], pad, k_slice.shape[2]), k_slice], dim=1)
                        v_slice = torch.cat([v_slice.new_zeros(v_slice.shape[0], pad, v_slice.shape[2]), v_slice], dim=1)
                    
                    k_list.append(k_slice)
                    v_list.append(v_slice)
                
                hidden_flatten.append((
                    torch.stack(k_list, dim=1).flatten(0, 1).detach(),
                    torch.stack(v_list, dim=1).flatten(0, 1).detach()
                ))
        else:
            hidden_flatten = None

        # is_firsts flatten (B, L) -> (B*L, 1)
        self.outer.detached_is_firsts = f.flatten(start_dim=0, end_dim=1).unsqueeze(dim=1).detach()

        # is_firsts hidden flatten (B, C+L) -> (B*L, C)
        if1_list = []
        for t in range(L):
            # Check if is_firsts_hidden_concat has enough length
            if is_firsts_hidden_concat.shape[1] >= L:
                end_idx = is_firsts_hidden_concat.shape[1] - L + t
                start_idx = max(0, end_idx - C)
                if1_slice = is_firsts_hidden_concat[:, start_idx:end_idx]
            else:
                # Fallback for short sequences
                if1_slice = is_firsts_hidden_concat[:, :t]
            
            if if1_slice.shape[1] < C:
                pad = C - if1_slice.shape[1]
                if1_slice = torch.cat([if1_slice.new_ones(if1_slice.shape[0], pad), if1_slice], dim=1)
            if1_list.append(if1_slice)
        
        self.outer.detached_is_firsts_hidden = torch.stack(if1_list, dim=1).flatten(0, 1).detach()

        # Flatten and detach post (B, L, D) -> (B*L, 1, D) = (B', 1, D)
        self.outer.detached_posts = {k: hidden_flatten if k == "hidden" else v.flatten(start_dim=0, end_dim=1).unsqueeze(dim=1).detach() if v is not None else None for k, v in posts.items()}
        return losses["total"]
