"""CEM Planner (Cross-Entropy Method).

From V-JEPA2 + LeWM.
Samples action trajectories, selects elite, updates distribution.
Momentum-based smoothing for stable planning.

Reference: V-JEPA2 mpc_utils.py, LeWM eval.py
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Callable
from torch_wm.utils import get_logger

from .base_planner import BasePlanner

logger = get_logger(__name__)

class CEMPlanner(BasePlanner):
    """
    Cross-Entropy Method planner.
    
    Iteratively samples action trajectories,
    selects top-k (elite), updates sampling distribution.
    
    Config params:
        - samples: int - number of samples per iteration (default: 100)
        - topk: int - number of elite samples (default: 10)
        - iterations: int - CEM iterations (default: 100)
        - momentum_mean: float - EMA for mean update (default: 0.25)
        - momentum_std: float - EMA for std update (default: 0.95)
        - init_std: float - initial action std (default: 0.1)
        - action_min: float - minimum action (default: -1.0)
        - action_max: float - maximum action (default: 1.0)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.samples = config.get('samples', 100)
        self.topk = config.get('topk', 10)
        self.iterations = config.get('iterations', 100)
        self.momentum_mean = config.get('momentum_mean', 0.25)
        self.momentum_std = config.get('momentum_std', 0.95)
        self.init_std = config.get('init_std', 0.1)
        self.action_min = config.get('action_min', -1.0)
        self.action_max = config.get('action_max', 1.0)
    
    def plan(
        self,
        current_latent: torch.Tensor,
        world_model: Callable,
        goal_latent: Optional[torch.Tensor] = None,
        horizon: Optional[int] = None
    ) -> torch.Tensor:
        """
        Plan action sequence using CEM.
        
        Args:
            current_latent: (B, D)
            world_model: Function (latent, actions) -> final_latent OR total_reward
            goal_latent: (B, D) optional
            horizon: Planning horizon
            
        Returns:
            Actions: (B, horizon, action_dim)
        """
        h = horizon or self.horizon
        B = current_latent.shape[0]
        device = current_latent.device
        
        # Initialize distribution
        mean = torch.zeros(B, h, self.action_dim, device=device)
        std = torch.ones(B, h, self.action_dim, device=device) * self.init_std
        
        # CEM iterations
        for iteration in range(self.iterations):
            # Sample action trajectories
            actions = mean.unsqueeze(0) + std.unsqueeze(0) * \
                      torch.randn(self.samples, B, h, self.action_dim, device=device)
            
            # Clip actions
            actions = torch.clamp(actions, self.action_min, self.action_max)
            
            # Evaluate samples
            costs = []
            for i in range(self.samples):
                # Rollout through world model
                output = world_model(current_latent, actions[i])
                
                if goal_latent is not None:
                    # Cost-based: distance to goal
                    if isinstance(output, tuple):
                        final_latent = output[0]
                    else:
                        final_latent = output
                    # final_latent might be (B, D) or (B, 1, D)
                    cost = F.mse_loss(final_latent, goal_latent, reduction='none')
                    # Sum over all dims except batch
                    cost = cost.view(B, -1).sum(dim=-1)
                else:
                    # Reward-based: maximize rewards (cost = -reward)
                    # output is expected to be total cumulative reward (B,) or scalar
                    cost = -output
                    if isinstance(cost, torch.Tensor):
                        cost = cost.view(B)
                
                costs.append(cost)
            
            costs = torch.stack(costs, dim=0)  # (samples, B)
            
            # Select elite
            _, elite_idx = torch.topk(costs, k=self.topk, dim=0, largest=False)
            
            # Expand elite_idx for all B: (topk, B) -> (topk, B, h, A)
            elite_idx_expanded = elite_idx.view(self.topk, B, 1, 1).expand(-1, -1, h, self.action_dim)
            elite_actions = torch.gather(actions, 0, elite_idx_expanded)  # (topk, B, h, A)
            
            # Update distribution with momentum
            elite_mean = elite_actions.mean(dim=0)
            elite_std = elite_actions.std(dim=0)
            
            mean = self.momentum_mean * elite_mean + (1 - self.momentum_mean) * mean
            std = self.momentum_std * elite_std + (1 - self.momentum_std) * std
        
        # Update metrics
        self._metrics['cem_best_cost'] = costs.min().item()
        self._metrics['cem_mean_cost'] = costs.mean().item()
        
        return mean
    
    def reset(self) -> None:
        """Reset planner."""
        self._metrics.clear()
