"""Gradient Descent Planner.

From temporal-straightening repo.
Optimizes actions via gradient descent on latent distance to goal.
Works best with straightened latents.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Callable
from torch_wm.utils import get_logger

from .base_planner import BasePlanner

logger = get_logger(__name__)

class GDPlanner(BasePlanner):
    """
    Gradient-based action optimizer.
    
    Optimizes action sequence to minimize distance to goal.
    Requires differentiable world model.
    
    Config params:
        - lr: float - learning rate (default: 0.1)
        - iterations: int - optimization steps (default: 30)
        - action_min: float - min action (default: -1.0)
        - action_max: float - max action (default: 1.0)
        - curvature_weight: float - straightening regularization (default: 0.1)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.lr = config.get('lr', 0.1)
        self.iterations = config.get('iterations', 30)
        self.action_min = config.get('action_min', -1.0)
        self.action_max = config.get('action_max', 1.0)
        self.curvature_weight = config.get('curvature_weight', 0.1)
    
    def plan(
        self,
        current_latent: torch.Tensor,
        world_model: Callable,
        goal_latent: Optional[torch.Tensor] = None,
        horizon: Optional[int] = None
    ) -> torch.Tensor:
        """
        Optimize actions via gradient descent.
        
        Args:
            current_latent: (B, D)
            world_model: Function (latent, actions) -> (final_latent, trajectory)
            goal_latent: (B, D) optional
            horizon: Planning horizon
            
        Returns:
            Actions: (B, horizon, action_dim)
        """
        if goal_latent is None:
            raise ValueError("GDPlanner requires a goal_latent to optimize towards.")
            
        h = horizon or self.horizon
        B = current_latent.shape[0]
        device = current_latent.device
        
        # Initialize actions
        actions = torch.zeros(B, h, self.action_dim, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([actions], lr=self.lr)
        
        # Optimization loop
        for iteration in range(self.iterations):
            # Forward through world model
            final_latent, trajectory = world_model(current_latent, actions)
            
            # Distance to goal
            dist_loss = F.mse_loss(final_latent, goal_latent)
            
            # Curvature regularization (straightening)
            curvature = self._compute_curvature(trajectory)
            
            # Combined loss
            loss = dist_loss + self.curvature_weight * curvature
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([actions], max_norm=1.0)
            
            optimizer.step()
            
            # Project actions to valid range
            with torch.no_grad():
                actions.clamp_(self.action_min, self.action_max)
        
        # Update metrics
        self._metrics['gd_final_dist'] = dist_loss.item()
        self._metrics['gd_curvature'] = curvature.item()
        
        return actions.detach()
    
    def _compute_curvature(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Compute trajectory curvature for regularization."""
        if trajectory.shape[1] < 3:
            return torch.tensor(0.0, device=trajectory.device)
        
        v1 = trajectory[:, 1:-1] - trajectory[:, :-2]
        v2 = trajectory[:, 2:] - trajectory[:, 1:-1]
        
        cos_sim = F.cosine_similarity(v1, v2, dim=-1)
        curvature = (1.0 - cos_sim).mean()
        
        return curvature
    
    def reset(self) -> None:
        """Reset planner."""
        self._metrics.clear()
