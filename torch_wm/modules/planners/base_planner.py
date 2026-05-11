"""Base Planner Interface.

Abstract interface for all planning algorithms.
Follows Strategy Pattern - interchangeable planners.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import torch
from torch_wm.utils import get_logger

logger = get_logger(__name__)

class BasePlanner(ABC):
    """
    Abstract interface for planning algorithms.
    
    All planners must implement:
    - plan(): Generate action sequence
    - reset(): Reset internal state
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize planner.
        
        Args:
            config: Planner configuration
        """
        self.config = config
        self.horizon = config.get('horizon', 5)
        self.action_dim = config.get('action_dim', 2)  # steer, accel for CARLA
        self._metrics: Dict[str, float] = {}
    
    @abstractmethod
    def plan(
        self,
        current_latent: torch.Tensor,
        world_model: Any,
        goal_latent: Optional[torch.Tensor] = None,
        horizon: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate action sequence.
        
        Args:
            current_latent: (B, D) current state
            world_model: World model for rollout
            goal_latent: (B, D) optional goal state
            horizon: Planning horizon (override config)
            
        Returns:
            Actions: (B, horizon, action_dim)
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset planner internal state."""
        pass
    
    def get_metrics(self) -> Dict[str, float]:
        """Get planner metrics."""
        return self._metrics.copy()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(horizon={self.horizon})"
