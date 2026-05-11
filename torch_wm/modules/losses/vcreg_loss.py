import torch
import torch.nn.functional as F
from torch_wm.core.base import BaseLoss
from torch_wm.core.registry import ModuleRegistry

@ModuleRegistry.register('vcreg')
class VCRegLoss(BaseLoss):
    """
    Variance-Covariance Regularization (part of VICReg).
    
    Helps prevent informational collapse by:
    - Variance: Forcing each dimension to have a minimum standard deviation.
    - Covariance: Forcing different dimensions to be decorrelated.
    """
    
    def __init__(self, config, weight=1.0):
        super().__init__(config, weight)
        self.var_weight = config.get('var_weight', 1.0)
        self.cov_weight = config.get('cov_weight', 1.0)
        self.eps = 1e-4

    def compute(self, model_outputs, batch, **kwargs):
        # We regularize the deterministic features of the posterior
        post = model_outputs.get("posts", model_outputs.get("post", {}))
        x = post.get("deter") # (B, L, D) or (B, D)
        
        if x is None:
            return torch.tensor(0.0)
            
        # Flatten time if present
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
            
        B, D = x.shape
        if B <= 1:
            return torch.tensor(0.0, device=x.device)

        # 1. Variance Loss
        std = torch.sqrt(x.var(dim=0) + self.eps)
        var_loss = torch.mean(F.relu(1 - std))
        
        # 2. Covariance Loss
        # Center the features
        x = x - x.mean(dim=0)
        # Compute covariance matrix
        cov = (x.T @ x) / (B - 1)
        # Clear diagonal
        mask = ~torch.eye(D, device=x.device).bool()
        cov_loss = (cov[mask]**2).sum() / D
        
        total_loss = self.var_weight * var_loss + self.cov_weight * cov_loss
        
        self._metrics["loss_vcreg_var"] = var_loss.item()
        self._metrics["loss_vcreg_cov"] = cov_loss.item()
        
        return total_loss

    def name(self) -> str:
        return "vcreg"
