import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_wm.core.base import BaseLoss
from torch_wm.core.registry import ModuleRegistry

@ModuleRegistry.register('joint_embedding')
class JointEmbeddingLoss(BaseLoss):
    """
    Joint-Embedding Predictive Architecture (JEPA) Loss.
    
    Instead of reconstructing pixels, aligns the predicted latent state 
    with the actual encoded state from the environment.
    
    Supports:
    - MSE alignment with learnable projection heads
    - Variance/Invariance/Covariance regularization (VICReg)
    """
    
    def __init__(self, config, weight=1.0):
        super().__init__(config, weight)
        self.sim_weight = config.get('sim_weight', 1.0)
        self.var_weight = config.get('var_weight', 1.0)
        self.cov_weight = config.get('cov_weight', 0.1)
        
        self.latent_dim = config.get('latent_dim', 512) # Space for alignment
        
        # Projectors for joint embedding space (using LazyLinear for automatic dim matching)
        self.predictor_projector = nn.Sequential(
            nn.LazyLinear(self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim)
        )
        
        self.target_projector = nn.Sequential(
            nn.LazyLinear(self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim)
        )

    def compute(self, model_outputs, batch, **kwargs):
        # post: posterior latent from online encoder (used for targets if EMA not present)
        # prior: predicted latent from dynamics (priors)
        prior = model_outputs.get("priors", model_outputs.get("prior", {}))
        
        if not prior:
            return torch.tensor(0.0, device=next(self.parameters()).device if list(self.parameters()) else "cpu")
            
        # Target representation (z_target)
        # 1. Try EMA encoder if available
        ema_encoder = kwargs.get("ema_encoder")
        if ema_encoder is not None and "states" in batch:
            with torch.no_grad():
                ema_out = ema_encoder(batch["states"])
                z_target_raw = ema_out["latent"].detach()
        else:
            # Fallback: align with detached features of the online encoder
            z_target_raw = model_outputs.get("latent", model_outputs.get("feats")).detach()

        # Prediction (z_pred)
        # We need the features predicted by the dynamics (priors)
        tssm = kwargs.get("tssm")
        if tssm is not None:
            z_pred_raw = tssm.get_feat(prior)
        else:
            # If no TSSM, we might be in imagine mode or fallback
            z_pred_raw = model_outputs.get("feats") 
        
        if z_pred_raw is None or z_target_raw is None:
            return torch.tensor(0.0, device=z_target_raw.device if z_target_raw is not None else "cpu")

        # Handle sequence length mismatches (e.g. predictor starts from t=1)
        if z_pred_raw.shape[1] != z_target_raw.shape[1]:
            min_l = min(z_pred_raw.shape[1], z_target_raw.shape[1])
            z_pred_raw = z_pred_raw[:, :min_l]
            z_target_raw = z_target_raw[:, :min_l]

        # Project both to common latent space
        # Flatten B,L for linear layers
        B, L, _ = z_pred_raw.shape
        z_pred = self.predictor_projector(z_pred_raw.reshape(-1, z_pred_raw.shape[-1]))
        z_target = self.target_projector(z_target_raw.reshape(-1, z_target_raw.shape[-1]))

        # 1. Similarity Loss (Invariance)
        sim_loss = F.mse_loss(z_pred, z_target)
        
        # 2. Variance Regularization (Prevents collapse)
        # Compute std over the batch dimension
        std_z_pred = torch.sqrt(z_pred.var(dim=0) + 1e-04)
        std_z_target = torch.sqrt(z_target.var(dim=0) + 1e-04)
        var_loss = torch.mean(F.relu(1 - std_z_pred)) + torch.mean(F.relu(1 - std_z_target))
        
        # 3. Covariance Regularization (Prevents informational collapse)
        def covariance_loss(x):
            x = x - x.mean(dim=0)
            cov = (x.T @ x) / (B * L - 1)
            off_diag = cov.pow(2).sum() - cov.diag().pow(2).sum()
            return off_diag / x.shape[-1]
            
        cov_loss = covariance_loss(z_pred) + covariance_loss(z_target)
        
        total_loss = (self.sim_weight * sim_loss + 
                      self.var_weight * var_loss + 
                      self.cov_weight * cov_loss)
        
        self._metrics["loss_je_sim"] = sim_loss.item()
        self._metrics["loss_je_var"] = var_loss.item()
        self._metrics["loss_je_cov"] = cov_loss.item()
        
        return total_loss

    def name(self) -> str:
        return "joint_embedding"
