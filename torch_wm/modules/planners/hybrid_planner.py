"""Hybrid Planner (CEM + GD Refinement).

Combines CEM for coarse search with GD for local refinement.
Best of both worlds: global exploration + local optimization.
"""

import torch
from typing import Dict, Any, Optional, Callable
from torch_wm.utils import get_logger

from .base_planner import BasePlanner
from .cem_planner import CEMPlanner
from .gd_planner import GDPlanner

logger = get_logger(__name__)

class HybridPlanner(BasePlanner):
    """
    Hybrid planner: CEM for coarse + GD for refinement.
    
    Stage 1: CEM explores action space globally
    Stage 2: GD refines around CEM solution locally
    
    Config params:
        - cem_params: dict - CEM configuration
        - gd_params: dict - GD configuration
        - gd_lr: float - GD learning rate (default: 0.01)
        - gd_iterations: int - GD refinement steps
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Initialize sub-planners
        self.cem = CEMPlanner(config.get('cem_params', {}))
        self.gd = GDPlanner(config.get('gd_params', {}))
    
    def plan(
        self,
        current_latent: torch.Tensor,
        world_model: Callable,
        goal_latent: Optional[torch.Tensor] = None,
        horizon: Optional[int] = None
    ) -> torch.Tensor:
        """
        Plan with CEM + GD refinement.
        
        Args:
            current_latent: (B, D)
            world_model: Rollout function
            goal_latent: (B, D) optional
            horizon: Planning horizon
            
        Returns:
            Actions: (B, horizon, action_dim)
        """
        # Stage 1: CEM coarse plan
        coarse_actions = self.cem.plan(
            current_latent, world_model, goal_latent, horizon
        )
        
        # Stage 2: GD refinement (only if goal provided)
        if goal_latent is not None:
            refined_actions = self._gd_refinement(
                current_latent, goal_latent, coarse_actions, world_model, horizon
            )
            return refined_actions
        
        return coarse_actions
    
    def _gd_refinement(
        self,
        current_latent: torch.Tensor,
        goal_latent: torch.Tensor,
        initial_actions: torch.Tensor,
        world_model: Callable,
        horizon: Optional[int] = None
    ) -> torch.Tensor:
        """
        Refine actions with gradient descent.
        
        Args:
            current_latent: (B, D)
            goal_latent: (B, D)
            initial_actions: (B, h, A) from CEM
            world_model: Rollout function
            horizon: Planning horizon
            
        Returns:
            Refined actions: (B, h, A)
        """
        h = horizon or self.horizon
        device = current_latent.device
        
        # Start from CEM solution
        actions = initial_actions.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([actions], lr=self.gd.lr)
        
        # Local refinement (few iterations)
        for _ in range(self.gd.iterations):
            final_latent, trajectory = world_model(current_latent, actions)
            
            dist_loss = torch.nn.functional.mse_loss(final_latent, goal_latent)
            curvature = self.gd._compute_curvature(trajectory)
            
            loss = dist_loss + self.gd.curvature_weight * curvature
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([actions], max_norm=1.0)
            optimizer.step()
            
            with torch.no_grad():
                actions.clamp_(self.gd.action_min, self.gd.action_max)
        
        # Update metrics
        self._metrics['hybrid_cem_cost'] = self.cem.get_metrics().get('cem_mean_cost', 0.0)
        self._metrics['hybrid_gd_dist'] = dist_loss.item()
        
        return actions.detach()
    
    def reset(self) -> None:
        """Reset both planners."""
        self.cem.reset()
        self.gd.reset()
        self._metrics.clear()
