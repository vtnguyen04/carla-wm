"""CPC Loss Module (Contrastive Predictive Coding).

Implements contrastive predictive coding loss from WMAgent.
Predicts future states contrastively - discriminate positive from negative samples.

Reference: Oord et al. 2018 "Representation Learning with Contrastive Predictive Coding"
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any
from carla_env.toolkit.utils import get_logger

from torch_wm.core.registry import ModuleRegistry
from torch_wm.core.base import BaseLoss

logger = get_logger(log_dir=".", job_name=__name__)

@ModuleRegistry.register('cpc')
class CPCLoss(BaseLoss):
    """
    Contrastive Predictive Coding loss.

    Learns discriminative representations by predicting future states
    and contrasting them against negative samples.

    Config params:
        - contrastive_steps: Number of future steps to predict (default: 10)
        - exp_lambda: Exponential decay for multi-step (default: 0.75)
    """

    def name(self): return "cpc"

    def __init__(self, config: Dict[str, Any] = None, weight=1.0):
        super().__init__(weight=weight)
        if config is None: config = {}
        self.contrastive_steps = config.get('contrastive_steps', 10)
        self.exp_lambda = config.get('exp_lambda', 0.75)

    def compute(self, model_outputs: Dict[str, Any], batch: Dict[str, Any], **kwargs) -> torch.Tensor:
        """
        Compute CPC contrastive loss using the contrastive_network.
        
        Args:
            model_outputs: Dictionary containing 'feats' (projected features) and 'latent' (raw encoder outputs)
            batch: Dictionary containing the environment data
            tssm: The TSSM module
            contrastive_network: nn.ModuleList containing predictors for each step
        """
        feats = model_outputs.get("feats")   # (B, L, D_feat)
        latent = model_outputs.get("latent") # (B, L, D_latent)
        contrastive_network = kwargs.get("contrastive_network")
        
        if feats is None or latent is None or contrastive_network is None:
            # We don't warning here as CPC might be disabled or missing in pre-fill
            return torch.tensor(0.0, device=feats.device if feats is not None else torch.device("cpu"))

        B, L, _ = feats.shape
        total_loss = 0.0
        accurate_counts = 0
        total_pairs = 0
        
        # We predict k-steps ahead: feats[t] -> latent[t+k]
        num_nets = len(contrastive_network)
        config = kwargs.get("config", {})
        
        contrastive_offsets = getattr(config, "contrastive_offsets", None)
        if contrastive_offsets is not None:
            target_steps = list(contrastive_offsets)[:num_nets]
        else:
            target_steps = list(range(1, min(self.contrastive_steps, num_nets) + 1))
            
        for k_idx, k in enumerate(target_steps):
            if L <= k:
                break
                
            # Select t and t+k pairs
            # current_feats: (B, L-k, D_feat)
            # future_latent: (B, L-k, D_latent)
            current_feats = feats[:, :-k]
            future_latent = latent[:, k:]
            
            # Predict through k-th contrastive network
            # proj_feats: (B, L-k, D_proj), proj_latent: (B, L-k, D_proj)
            proj_feats, proj_latent = contrastive_network[k_idx](current_feats, future_latent)
            
            # InfoNCE Similarity
            # Normalized for cosine similarity if required, otherwise dot product
            proj_feats = F.normalize(proj_feats, dim=-1)
            proj_latent = F.normalize(proj_latent, dim=-1)
            
            # Similarity Matrix (B, L-k, L-k)
            # Row: prediction for step t, Column: actual latent at step t'
            logits = torch.bmm(proj_feats, proj_latent.transpose(1, 2)) # (B, L-k, L-k)
            
            # Temperature scaling (optional, WMAgent typically uses 1.0 or learns it)
            logits = logits / 1.0 
            
            # Target labels: diagonal elements are correct matches
            cur_L = L - k
            labels = torch.arange(cur_L, device=feats.device).unsqueeze(0).expand(B, -1)
            
            # Cross entropy loss
            step_loss = F.cross_entropy(logits.reshape(-1, cur_L), labels.reshape(-1))
            
            # Weighted by exponential decay
            weight = self.exp_lambda ** (k - 1)
            total_loss += weight * step_loss
            
            # Tracking accuracy
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                accurate_counts += (preds == labels).sum().item()
                total_pairs += B * cur_L
                
        # Metrics
        if total_pairs > 0:
            self.update_metrics({
                "cpc_accuracy": accurate_counts / total_pairs,
                "cpc_last_loss": total_loss.item()
            })
            
        return total_loss / self.contrastive_steps

    def reset(self) -> None:
        """Reset CPC loss state."""
        self._metrics.clear()

    def _compute_accuracy(
        self,
        sim_matrix: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor = None
    ) -> float:
        """
        Compute prediction accuracy for CPC.

        Args:
            sim_matrix: (B, L, L) similarity matrix
            labels: (B, L) target labels
            mask: Optional mask

        Returns:
            Accuracy as float
        """
        predictions = sim_matrix.argmax(dim=-1)
        correct = (predictions == labels).float()

        if mask is not None:
            correct = correct * mask

        return correct.mean().item()
