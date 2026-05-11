"""Straightening Regularizer.

From temporal-straightening repo.
Penalizes angular deviations between consecutive latent steps.
Makes latent trajectories locally linear for better planning.

Reference: "Temporal Straightening in Visual Representation Learning"
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
from torch_wm.utils import get_logger

from torch_wm.core.base import BaseLoss
from torch_wm.core.registry import ModuleRegistry

logger = get_logger(__name__)

@ModuleRegistry.register('straightening')
class StraighteningRegularizer(BaseLoss):
    """
    Temporal straightening regularizer.
    
    Penalizes curvature in latent trajectories.
    Makes Euclidean distance ≈ geodesic distance.
    Improves planning stability.
    
    Config params:
        - type: str - 'cos' or 'aggcos' (default: 'cos')
        - weight: float - regularization strength (default: 0.1)
        - eps: float - numerical stability (default: 1e-6)
    """
    
    def __init__(self, config: Dict[str, Any], weight: float = 1.0):
        super().__init__(config, weight)
        self.type = config.get('type', 'cos')
        self.eps = config.get('eps', 1e-6)
    
    def name(self) -> str:
        return 'straightening'

    def compute(self, model_outputs: Dict[str, Any], batch: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """
        Compute straightening loss.
        """
        # Get latents from post state
        if 'post' not in model_outputs or 'deter' not in model_outputs['post']:
            return torch.tensor(0.0, device=next(iter(model_outputs.values())).device if model_outputs else 'cpu')
            
        latents = model_outputs['post']['deter']
        
        if latents.shape[1] < 3:
            return torch.tensor(0.0, device=latents.device)
        
        if self.type == 'cos':
            loss = self._cos_curvature(latents)
        elif self.type == 'aggcos':
            loss = self._aggcos_curvature(latents)
        else:
            raise ValueError(f"Unknown type: {self.type}")
        
        self.update_metrics({'straightening_loss': loss.item()})
        
        return loss
    
    def _cos_curvature(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine curvature on raw features.
        
        v1 = features[t] - features[t-1]
        v2 = features[t+1] - features[t]
        loss = 1 - cos(v1, v2)
        """
        # Velocity vectors
        v1 = latents[:, 1:-1] - latents[:, :-2]
        v2 = latents[:, 2:] - latents[:, 1:-1]
        
        # Cosine similarity
        cos_sim = F.cosine_similarity(v1, v2, dim=-1, eps=self.eps)
        
        # Curvature = 1 - cos(theta)
        loss = (1.0 - cos_sim).mean()
        
        return loss
    
    def _aggcos_curvature(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine curvature on aggregated features.
        
        Same as cos but with mean-pooling over feature dimension first.
        More stable for high-dimensional latents.
        """
        # Pool over feature dimension
        latents_pooled = latents.mean(dim=-1)  # (B, L)
        
        # Velocity vectors
        v1 = latents_pooled[:, 1:-1] - latents_pooled[:, :-2]
        v2 = latents_pooled[:, 2:] - latents_pooled[:, 1:-1]
        
        # Normalize
        v1_norm = F.normalize(v1, dim=-1, eps=self.eps)
        v2_norm = F.normalize(v2, dim=-1, eps=self.eps)
        
        # Cosine similarity
        cos_sim = (v1_norm * v2_norm).sum(dim=-1)
        
        # Curvature
        loss = (1.0 - cos_sim).mean()
        
        return loss
