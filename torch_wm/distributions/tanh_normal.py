"""
Tanh Normal Distribution

Tanh-transformed Normal distribution for continuous actions.
Provides proper log_prob correction for tanh squashing.

Used for bounded continuous action spaces like CARLA steering/acceleration.
"""

import torch
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Optional

from torch_wm.distributions.base import BaseDist

class TanhNormal(BaseDist):
    """
    Tanh-transformed Normal distribution.
    
    Samples: tanh(Normal(mean, std))
    log_prob: adjusts for tanh Jacobian
    
    This ensures proper probability correction when actions
    are squashed through tanh for bounded action spaces.
    """
    
    def __init__(self, loc: torch.Tensor, scale: torch.Tensor, 
                 validate_args: bool = False):
        """
        Initialize TanhNormal distribution.
        
        Args:
            loc: Mean of base Normal distribution
            scale: Standard deviation of base Normal distribution
            validate_args: Whether to validate arguments
        """
        self.base_dist = Normal(loc, scale, validate_args=validate_args)
        self._loc = loc
        self._scale = scale
    
    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Reparameterized sample, then apply tanh.
        
        Args:
            sample_shape: Shape of samples
            
        Returns:
            Tanh-transformed samples in [-1, 1]
        """
        raw = self.base_dist.rsample(sample_shape)
        return torch.tanh(raw)
    
    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Non-reparameterized sample.
        
        Args:
            sample_shape: Shape of samples
            
        Returns:
            Tanh-transformed samples in [-1, 1]
        """
        raw = self.base_dist.sample(sample_shape)
        return torch.tanh(raw)
    
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Log probability of value under TanhNormal.
        
        Uses inverse tanh to get back to raw space, then adjusts for Jacobian.
        
        Args:
            value: Value to compute log_prob for (should be in [-1, 1])
            
        Returns:
            Log probability (corrected for tanh)
        """
        # Clamp to avoid numerical issues with atanh
        value = torch.clamp(value, -0.999, 0.999)
        
        # Inverse tanh to get raw value
        raw = torch.atanh(value)
        
        # Base distribution log_prob
        base_log_prob = self.base_dist.log_prob(raw)
        
        # Jacobian correction: sum of log(1 - tanh(x)^2) = sum of log(1 - value^2)
        # This accounts for the tanh transformation
        jac_correction = torch.log(1 - value ** 2 + 1e-6)
        
        return base_log_prob - jac_correction
    
    def mode(self) -> torch.Tensor:
        """
        Mode of tanh(Normal) = tanh(mean).
        
        Returns:
            Tanh of mean
        """
        return torch.tanh(self.base_dist.loc)
    
    def mean(self) -> torch.Tensor:
        """
        Approximate mean (not analytically tractable).
        
        Returns:
            Tanh of mean (approximation)
        """
        return torch.tanh(self.base_dist.loc)
    
    def entropy(self) -> torch.Tensor:
        """
        Approximate entropy.
        
        Uses base Normal entropy as approximation (not exact after tanh).
        
        Returns:
            Entropy approximation
        """
        # Approximation: use base Normal entropy
        # Not exact after tanh but works well in practice
        return self.base_dist.entropy()
    
    @property
    def loc(self) -> torch.Tensor:
        """Get mean parameter."""
        return self._loc
    
    @property
    def scale(self) -> torch.Tensor:
        """Get std parameter."""
        return self._scale
    
    def __getattr__(self, name: str):
        """Delegate to base distribution if attribute not found."""
        return getattr(self.base_dist, name)

class IndependentTanhNormal:
    """
    Independent TanhNormal distributions for each action dimension.
    
    Wraps TanhNormal with Independent to create multivariate distribution.
    """
    
    def __init__(self, loc: torch.Tensor, scale: torch.Tensor, 
                 reinterpreted_batch_ndims: int = 1):
        """
        Initialize Independent TanhNormal.
        
        Args:
            loc: Mean tensor (..., action_dim)
            scale: Scale tensor (..., action_dim)
            reinterpreted_batch_ndims: Number of dims to reinterpret
        """
        self.base = TanhNormal(loc, scale)
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims
        self._loc = loc
        self._scale = scale
    
    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Reparameterized sample."""
        return self.base.rsample(sample_shape)
    
    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Sample."""
        return self.base.sample(sample_shape)
    
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Log probability with proper reduction.
        
        Sums over action dimensions.
        """
        return self.base.log_prob(value).sum(dim=-1)
    
    def mode(self) -> torch.Tensor:
        """Mode."""
        return self.base.mode()
    
    def mean(self) -> torch.Tensor:
        """Mean."""
        return self.base.mean()
    
    def entropy(self) -> torch.Tensor:
        """Entropy approximation."""
        return self.base.entropy().sum(dim=-1)
