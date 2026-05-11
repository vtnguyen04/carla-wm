"""Curvature Loss Module (Temporal Straightening).

Implements temporal straightening from temporal-straightening repo.
Penalizes curvature in latent trajectories to encourage linear dynamics.
"""

from typing import Any, Dict

import torch
import torch.nn.functional as F

from torch_wm.core.base import BaseLoss
from torch_wm.core.registry import ModuleRegistry

@ModuleRegistry.register('curvature')
class CurvatureLoss(BaseLoss):
    """Temporal straightening curvature loss.

    Penalizes angular deviations between consecutive latent steps.
    """

    def name(self) -> str:
        return "curvature"

    def __init__(self, config=None, weight=1.0):
        super().__init__(config=config, weight=weight)
        self.kappa = self.config.get("kappa", 0.1)

    def compute(
        self,
        model_outputs: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """Compute temporal straightening curvature loss."""
        latents = model_outputs.get("latent")
        if latents is None or latents.shape[1] < 3:
            return torch.tensor(0.0, device=latents.device if latents is not None else "cpu")

        # Compute curvature based on type (default cosine)
        return self._compute_cosine_curvature(latents)

    def _compute_cosine_curvature(self, latents: torch.Tensor) -> torch.Tensor:
        """Compute patch-wise cosine curvature penalty."""
        # v1 = latents[t] - latents[t-1]
        # v2 = latents[t+1] - latents[t]
        v1 = latents[:, 1:-1] - latents[:, :-2]
        v2 = latents[:, 2:] - latents[:, 1:-1]

        v1_norm = F.normalize(v1, dim=-1)
        v2_norm = F.normalize(v2, dim=-1)

        cos_sim = (v1_norm * v2_norm).sum(dim=-1)
        curvature = 1.0 - cos_sim
        return curvature.mean()
