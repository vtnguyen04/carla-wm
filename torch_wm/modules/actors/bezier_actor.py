"""
Bezier Actor - Predicts Control Points for Trajectory Generation

Instead of predicting raw waypoints or direct controls, the actor predicts
4 control points that define a cubic Bezier curve. This guarantees:
- Smooth, continuous trajectories (C1 continuous)
- No zigzag or oscillation
- Natural driving behavior

The 4 control points are relative to the current vehicle position,
ensuring translation invariance and easier learning.
"""

import torch
import torch.nn as nn
import numpy as np

class BezierActor(nn.Module):
    """
    Predict 4 control points for a cubic Bezier trajectory.
    
    Control points are relative to current position:
    - P0: Start (near vehicle, small offset)
    - P1: First intermediate point
    - P2: Second intermediate point  
    - P3: End point (farthest ahead)
    
    The resulting Bezier curve is guaranteed to be smooth.
    """
    
    def __init__(
        self,
        feat_size: int,
        hidden_size: int = 256,
        num_control_points: int = 4,
        max_lookahead: float = 50.0,
        max_lateral: float = 10.0,
    ):
        """
        Args:
            feat_size: Input feature dimension from world model
            hidden_size: Hidden layer size
            num_control_points: Number of Bezier control points (default 4)
            max_lookahead: Maximum forward distance for control points (meters)
            max_lateral: Maximum lateral offset for control points (meters)
        """
        super().__init__()
        self.num_control_points = num_control_points
        self.max_lookahead = max_lookahead
        self.max_lateral = max_lateral
        
        # Control point spacing along the path
        # P0 at 5m, P1 at 15m, P2 at 30m, P3 at 50m ahead
        self.cp_forward_distances = torch.tensor([5.0, 15.0, 30.0, 50.0])
        
        # Network to predict lateral offsets for each control point
        self.network = nn.Sequential(
            nn.Linear(feat_size, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_control_points),  # Lateral offsets
        )
        
        # Network to predict speeds at each control point
        self.speed_head = nn.Sequential(
            nn.Linear(feat_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, num_control_points),  # Speeds
        )
    
    def forward(self, feat: torch.Tensor) -> dict:
        """
        Predict Bezier control points and speed profile.
        
        Args:
            feat: [B, feat_size] - Features from world model
        
        Returns:
            dict with:
                - control_points: [B, 4, 2] (x, y) relative to vehicle
                - speeds: [B, 4] - Target speeds at each control point
        """
        # Predict lateral offsets
        lateral_offsets = self.network(feat)  # [B, 4]
        
        # Clamp to valid range
        lateral_offsets = torch.tanh(lateral_offsets) * self.max_lateral
        
        # Construct control points:
        # x = forward distance (fixed), y = lateral offset (predicted)
        B = feat.shape[0]
        forward_dists = self.cp_forward_distances.to(feat.device).unsqueeze(0).expand(B, -1)
        
        # Control points in vehicle frame: (forward, lateral)
        control_points = torch.stack([forward_dists, lateral_offsets], dim=-1)  # [B, 4, 2]
        
        # Predict speeds at each control point
        speed_logits = self.speed_head(feat)  # [B, 4]
        speeds = torch.sigmoid(speed_logits) * 20.0  # 0-20 m/s (0-72 km/h)
        
        return {
            'control_points': control_points,
            'speeds': speeds,
            'lateral_offsets': lateral_offsets,
        }
    
    def get_trajectory(
        self,
        feat: torch.Tensor,
        num_samples: int = 30,
        bezier_generator=None,
    ) -> dict:
        """
        Get full Bezier trajectory from features.
        
        Args:
            feat: [B, feat_size]
            num_samples: Number of waypoints to sample
            bezier_generator: BezierTrajectoryGenerator instance (or creates one)
        
        Returns:
            dict with:
                - trajectory: [B, num_samples, 2] - (x, y) waypoints
                - control_points: [B, 4, 2] - Original control points
                - speeds: [B, 4] - Speed profile
        """
        from torch_wm.modules.trajectory import BezierTrajectoryGenerator
        
        # Predict control points
        pred = self.forward(feat)
        control_points = pred['control_points']  # [B, 4, 2]
        
        # Generate smooth trajectory
        if bezier_generator is None:
            bezier_generator = BezierTrajectoryGenerator(num_samples=num_samples)
        
        trajectory = bezier_generator.generate(control_points.detach().cpu().numpy())
        
        return {
            'trajectory': trajectory,
            'control_points': control_points,
            'speeds': pred['speeds'],
        }
