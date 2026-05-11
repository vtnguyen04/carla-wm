"""WMAgent Loss Modules.

Exposes all loss functions available for the WMAgent model.
"""

from .cpc_loss import CPCLoss
from .discount_loss import DiscountLoss
from .kl_balancing_loss import KLBalancingLoss
from .kl_loss import KLLoss
from .reconstruction_loss import ReconstructionLoss
from .reward_loss import RewardLoss

__all__ = [
    'CPCLoss',
    'DiscountLoss',
    'KLBalancingLoss',
    'KLLoss',
    'ReconstructionLoss',
    'RewardLoss',
]
