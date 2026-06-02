"""VCREG Regularizer (VICReg-inspired).

Prevents latent collapse and encourages diverse feature representations.
From temporal-straightening repo.

Standard deviation loss: Encourage each dimension to have std >= 1
Covariance loss: Decorrelate latent dimensions
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any
from torch_wm.utils import get_logger

from torch_wm.core.base import BaseLoss
from torch_wm.core.registry import ModuleRegistry

logger = get_logger(__name__)

@ModuleRegistry.register('vcreg')
class VCREGRegularizer(BaseLoss):
    """
    VCREG (Variance-Covariance Regularization).
    
    Prevents dimensional collapse in latent space.
    Encourages diverse, decorrelated features.
    
    Config params:
        - std_weight: float - std loss weight (default: 1.0)
        - cov_weight: float - cov loss weight (default: 1.0)
        - std_target: float - target std (default: 1.0)
    """
    
    def __init__(self, config: Dict[str, Any], weight: float = 1.0):
        super().__init__(config, weight)
        self.std_weight = config.get('std_weight', 1.0)
        self.cov_weight = config.get('cov_weight', 1.0)
        self.std_target = config.get('std_target', 1.0)
    
    def name(self) -> str:
        return 'vcreg'

    def compute(self, model_outputs: Dict[str, Any], batch: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """
        Compute VCREG loss.
        """
        # Get latents from post state
        if 'post' not in model_outputs or 'stoch' not in model_outputs['post']:
            for value in model_outputs.values():
                if isinstance(value, torch.Tensor):
                    return torch.tensor(0.0, device=value.device)
            return torch.tensor(0.0)
            
        latents = model_outputs['post']['stoch']
        
        # Flatten to (N, D)
        if latents.dim() >= 3:
            latents = latents.reshape(-1, latents.shape[-1])
        
        # Standard deviation loss
        std_loss = self._std_loss(latents)
        
        # Covariance loss
        cov_loss = self._cov_loss(latents)
        
        # Combined
        total_loss = self.std_weight * std_loss + self.cov_weight * cov_loss
        
        self.update_metrics({
            'vcreg_std_loss': std_loss.item(),
            'vcreg_cov_loss': cov_loss.item(),
            'vcreg_total_loss': total_loss.item()
        })
        
        return total_loss
    
    def _std_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encourage each latent dimension to have std >= target.
        
        Uses soft constraint: ReLU(target - std)
        """
        std = torch.sqrt(x.var(dim=0) + 1e-4)
        loss = F.relu(self.std_target - std).mean()
        return loss
    
    def _cov_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decorrelate latent dimensions.
        
        Off-diagonal elements of covariance matrix should be zero.
        """
        # Center
        x = x - x.mean(dim=0)
        
        # Covariance matrix
        cov = (x.T @ x) / (x.shape[0] - 1)
        
        # Off-diagonal elements
        d = cov.shape[0]
        mask = ~torch.eye(d, dtype=bool, device=cov.device)
        
        # Sum of squared off-diagonal elements
        loss = cov[mask].pow(2).sum() / d
        
        return loss
