"""Reconstruction Loss Module.

Implements decoder reconstruction loss for world model training.
Measures how well the decoder can reconstruct observations from latent representations.

Supports multiple reconstruction targets:
- Image reconstruction (cnn_keys)
- Scalar reconstruction
- Combined multi-modal reconstruction
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, List

from torch_wm.core.registry import ModuleRegistry
from torch_wm.core.base import BaseLoss
from carla_env.toolkit.utils import get_logger

logger = get_logger(log_dir=".", job_name=__name__)

@ModuleRegistry.register('reconstruction')
class ReconstructionLoss(BaseLoss):
    """
    Decoder reconstruction loss.

    Measures reconstruction quality for observations.
    Supports multiple modalities (images, scalars).

    Config params:
        - cnn_keys: List[str] - keys for image reconstruction (default: [])
        - scalar_keys: List[str] - keys for scalar reconstruction (default: [])
        - loss_type: str - 'mse' or 'bce' (default: 'mse')
    """
    
    def name(self): return "reconstruction"

    def __init__(self, config: Dict[str, Any] = None, weight=1.0):
        """
        Initialize reconstruction loss.
        """
        super().__init__(weight=weight)
        if config is None: config = {}
        self.cnn_keys = config.get('cnn_keys', [])
        self.scalar_keys = config.get('scalar_keys', [])
        self.loss_type = config.get('loss_type', 'mse')

    def compute(self, model_outputs, batch, **kwargs):
        """
        Compute reconstruction loss.
        """
        reconstructions = model_outputs.get("states_rec_dist")
        targets = batch.get("states")
        
        if reconstructions is None or targets is None:
            # Silent return during pre-fill if data is partial
            return torch.tensor(0.0)

        total_loss = 0.0
        num_keys = 0

        # Case 1: Reconstructions is a dictionary (Multi-Modal)
        if isinstance(reconstructions, dict):
            # Try configured keys first
            combined_keys = self.cnn_keys + self.scalar_keys
            
            for key in combined_keys:
                if key in reconstructions and key in targets:
                    recon_dist = reconstructions[key]
                    target = targets[key]
                    # MSE Dist log_prob handles (B, L, C, H, W) or (B, L, D) properly
                    key_loss = -recon_dist.log_prob(target.detach()).mean()
                    total_loss = total_loss + key_loss
                    num_keys += 1
            
            # Fallback: if no configured keys found or num_keys is 0, try all reconstruction keys
            if num_keys == 0:
                for key in reconstructions.keys():
                    if key in targets:
                        total_loss += -reconstructions[key].log_prob(targets[key].detach()).mean()
                        num_keys += 1
        
        # Case 2: Legacy Single-Modal (Distribution object)
        else:
            total_loss = -reconstructions.log_prob(targets.detach()).mean()
            num_keys = 1
                    
        if num_keys == 0:
            # If we expected reconstructions but found no matching targets, log a warning
            logger.warning(f"Reconstruction loss: matched 0/{{len(reconstructions)}} keys in targets. Targets keys: {{list(targets.keys()) if isinstance(targets, dict) else 'not a dict'}}")
            if isinstance(reconstructions, dict) and len(reconstructions) > 0:
                device = next(iter(reconstructions.values())).device
            elif not isinstance(reconstructions, dict):
                device = reconstructions.device
            else:
                device = "cpu"
            total_loss = torch.tensor(0.0, device=device)
            
        return total_loss

    def reset(self) -> None:
        """Reset reconstruction loss state."""
        self._metrics.clear()
