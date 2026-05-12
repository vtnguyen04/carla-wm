"""SIGReg Loss Module (from LE-WM).

Implements SIGReg from LE-WM (Latent-Embedding World Models) using the Epps-Pulley test statistic.
Prevents latent space collapse by enforcing Gaussian distribution across random projections.
"""

from typing import Any, Dict
import torch
import torch.nn as nn

from torch_wm.core.base import BaseLoss
from torch_wm.core.registry import ModuleRegistry

@ModuleRegistry.register('sigreg')
class SIGRegLoss(BaseLoss):
    """Sketch Isotropic Gaussian Regularizer using Epps-Pulley statistic.

    Reference: LE-WM (lucas-maes/le-wm)
    """

    def __init__(self, config=None, weight=1.0):
        super().__init__(config=config, weight=weight)
        self.knots = self.config.get("knots", 17)
        self.num_proj = self.config.get("num_proj", 1024)

        t = torch.linspace(0, 3, self.knots, dtype=torch.float32)
        dt = 3 / (self.knots - 1)
        weights = torch.full((self.knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        
        self.register_buffer("t_vec", t)
        self.register_buffer("phi", window)
        self.register_buffer("ep_weights", weights * window)

    def compute(
        self,
        model_outputs: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """Compute SIGReg loss on latent states using Epps-Pulley statistic."""
        # Align on deterministic feats or stoch states
        proj = model_outputs.get("feats", model_outputs.get("latent"))
        if proj is None:
            return torch.tensor(0.0, device=next(self.parameters()).device if list(self.parameters()) else "cpu")

        # latents is typically (B, L, D) in carTwister.
        # le-wm expects proj to be (T, B, D). Let's reshape to (L, B, D).
        if proj.dim() == 3:
            # (B, L, D) -> (L, B, D)
            proj = proj.transpose(0, 1)
        else:
            # (B, D) -> (1, B, D)
            proj = proj.unsqueeze(0)

        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t_vec
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.ep_weights) * proj.size(-2)
        
        loss = statistic.mean() * self.weight
        
        return loss

    def name(self) -> str:
        return "sigreg"
