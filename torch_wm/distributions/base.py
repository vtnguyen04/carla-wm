"""Base Distribution Interface.

Provides a common contract for all distribution classes in torch_wm.
Follows Liskov Substitution Principle — any BaseDist subclass can be used
interchangeably wherever a distribution is expected.

All distributions must implement:
    - mode(): Return the most likely value
    - log_prob(value): Return log probability of value
"""

from abc import ABC, abstractmethod
import torch


class BaseDist(ABC):
    """Abstract base class for all torch_wm distributions.
    
    Ensures a consistent interface for:
    - Decoder outputs (LaplaceDist, MSEDist)
    - Reward/Value heads (SymLogDiscreteDist)
    - Policy outputs (OneHotDist, TanhNormal)
    - Continue heads (Bernoulli)
    """

    @abstractmethod
    def mode(self) -> torch.Tensor:
        """Return the most likely value (MAP estimate)."""
        ...

    @abstractmethod
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Return log probability of the given value.
        
        Args:
            value: Target tensor to compute log probability for.
            
        Returns:
            Log probability tensor (negative for loss computation).
        """
        ...
