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

import torch
import warnings

# Lazy load LPIPS to avoid overhead / crash if not installed
_LOSS_VGG = None

class PerceptualLPIPSDist:
    """Perceptual LPIPS Distribution (VGG based).

    Acts as a probability distribution where negative log-likelihood
    is the LPIPS distance + optionally combined with L1 (Laplace).
    """
    def __init__(self, mode, include_l1=True, l1_weight=1.0, lpips_weight=1.0, agg="sum", reinterpreted_batch_ndims=0):
        global _LOSS_VGG
        if _LOSS_VGG is None:
            import lpips
            # Suppress torchvision warnings during init
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Sử dụng SqueezeNet siêu nhẹ thay vì VGG khổng lồ để tiết kiệm 90% VRAM
                _LOSS_VGG = lpips.LPIPS(net='squeeze', verbose=False).to(mode.device)
                _LOSS_VGG.eval()
                # Freeze
                for p in _LOSS_VGG.parameters():
                    p.requires_grad = False

        self._mode = mode
        self.include_l1 = include_l1
        self.l1_weight = l1_weight
        self.lpips_weight = lpips_weight

        self.agg = agg
        self.reduce_dims = tuple([-x for x in range(1, reinterpreted_batch_ndims + 1)])

    def mode(self):
        return self._mode

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)

        total_loss = 0.0

        # 1. L1 (Laplace) Term
        if self.include_l1:
            l1_dist = torch.abs(self._mode - value)
            if self.agg == "mean":
                l1_loss = l1_dist.mean(dim=self.reduce_dims)
            else:
                l1_loss = l1_dist.sum(dim=self.reduce_dims)
            total_loss += self.l1_weight * l1_loss

        # 2. LPIPS Term
        # Model features (logits) can be unbounded. SqueezeNet forward pass will overflow to NaN if fed extreme values.
        # We clamp to [-0.5, 0.5]. Outside this range, LPIPS provides 0 gradient, BUT the L1 loss above will naturally
        # and robustly pull the unbounded logits back into the valid range!
        mode_scaled = torch.clamp(self._mode, -0.5, 0.5) * 2.0
        value_scaled = value * 2.0

        # Flatten batch dimensions since LPIPS expects (B, C, H, W)
        flat_mode = mode_scaled.flatten(0, -4) if mode_scaled.dim() > 4 else mode_scaled
        flat_val  = value_scaled.flatten(0, -4) if value_scaled.dim() > 4 else value_scaled

        # Compute LPIPS
        # Note: LPIPS returns (N, 1, 1, 1) usually.
        lpips_dist = _LOSS_VGG(flat_mode, flat_val)

        # Reshape back to batch dims (e.g. B, L, 1, 1, 1 -> B, L)
        lpips_dist = lpips_dist.view(*self._mode.shape[:-3])

        # Match gradient scaling to the agg type
        # If sum aggregation, we implicitly multiply the LPIPS scalar by the number of pixels
        # so that it balances cleanly with the L1 sum loss
        if self.agg == "sum":
            # Per-pixel equivalent scaling to match typical L1 magnitude scaling
            num_pixels = self._mode.shape[-3] * self._mode.shape[-2] * self._mode.shape[-1]
            lpips_loss = lpips_dist * num_pixels
        else:
            lpips_loss = lpips_dist
            
        # Ensure scalar if total_loss is scalar (dim 0)
        if total_loss.dim() == 0 and lpips_loss.numel() == 1:
            lpips_loss = lpips_loss.squeeze()

        total_loss += self.lpips_weight * lpips_loss

        return - total_loss
