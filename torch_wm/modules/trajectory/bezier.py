"""
Bezier Trajectory Generator

Cubic Bezier curves from 4 control points.
Guarantees smooth, continuous trajectories - NO zigzag!

B(t) = (1-t)^3*P0 + 3*(1-t)^2*t*P1 + 3*(1-t)*t^2*P2 + t^3*P3
"""

import torch
import numpy as np
from typing import Union

class BezierTrajectoryGenerator:
    """
    Generate smooth trajectories from 4 control points using cubic Bezier curves.
    
    Properties:
    - C1 continuous (smooth first derivative)
    - Always stays within convex hull of control points
    - No oscillations or zigzag behavior
    """
    
    def __init__(self, num_samples: int = 30):
        """
        Args:
            num_samples: Number of waypoints to sample along the curve
        """
        self.num_samples = num_samples
        # Precompute t values
        self.t = torch.linspace(0, 1, num_samples)
    
    def generate(
        self,
        control_points: Union[np.ndarray, torch.Tensor],
        num_samples: int = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Generate Bezier trajectory from control points.
        
        Args:
            control_points: Shape (4, 2) or (B, 4, 2) - 4 control points (x, y)
            num_samples: Override number of samples
        
        Returns:
            trajectory: Shape (num_samples, 2) or (B, num_samples, 2)
        """
        n_samples = num_samples or self.num_samples
        
        if isinstance(control_points, np.ndarray):
            return self._generate_numpy(control_points, n_samples)
        else:
            return self._generate_torch(control_points, n_samples)
    
    def _generate_numpy(self, control_points: np.ndarray, n_samples: int) -> np.ndarray:
        """Generate trajectory using NumPy."""
        t = np.linspace(0, 1, n_samples)
        
        # Handle single trajectory (4, 2) or batch (B, 4, 2)
        if control_points.ndim == 2:
            P0, P1, P2, P3 = control_points[0], control_points[1], control_points[2], control_points[3]
            # (n_samples,) × (2,) → (n_samples, 2)
            trajectory = (
                (1-t)[:, None]**3 * P0[None, :] +
                3 * (1-t)[:, None]**2 * t[:, None] * P1[None, :] +
                3 * (1-t)[:, None] * t[:, None]**2 * P2[None, :] +
                t[:, None]**3 * P3[None, :]
            )
        else:
            # Batched: (B, 4, 2)
            B = control_points.shape[0]
            P0 = control_points[:, 0, :].reshape(B, 1, 2)  # (B, 1, 2)
            P1 = control_points[:, 1, :].reshape(B, 1, 2)
            P2 = control_points[:, 2, :].reshape(B, 1, 2)
            P3 = control_points[:, 3, :].reshape(B, 1, 2)
            
            t_t = t.reshape(1, n_samples, 1)  # (1, n_samples, 1)
            
            trajectory = (
                (1-t_t)**3 * P0 +
                3 * (1-t_t)**2 * t_t * P1 +
                3 * (1-t_t) * t_t**2 * P2 +
                t_t**3 * P3
            )
            # (B, n_samples, 2)
        
        return trajectory
    
    def _generate_torch(self, control_points: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Generate trajectory using PyTorch."""
        t = torch.linspace(0, 1, n_samples, device=control_points.device)
        
        if control_points.ndim == 2:
            P0, P1, P2, P3 = control_points[0], control_points[1], control_points[2], control_points[3]
            trajectory = (
                (1-t)[:, None]**3 * P0[None, :] +
                3 * (1-t)[:, None]**2 * t[:, None] * P1[None, :] +
                3 * (1-t)[:, None] * t[:, None]**2 * P2[None, :] +
                t[:, None]**3 * P3[None, :]
            )
        else:
            # Batched: (B, 4, 2)
            B = control_points.shape[0]
            P0 = control_points[:, 0, :].reshape(B, 1, 2)
            P1 = control_points[:, 1, :].reshape(B, 1, 2)
            P2 = control_points[:, 2, :].reshape(B, 1, 2)
            P3 = control_points[:, 3, :].reshape(B, 1, 2)
            
            t_t = t.reshape(1, n_samples, 1)
            
            trajectory = (
                (1-t_t)**3 * P0 +
                3 * (1-t_t)**2 * t_t * P1 +
                3 * (1-t_t) * t_t**2 * P2 +
                t_t**3 * P3
            )
        
        return trajectory
    
    def compute_curvature(self, trajectory: np.ndarray) -> float:
        """
        Compute total curvature of trajectory.
        Lower = smoother.
        
        curvature = mean(1 - cos(angle between consecutive segments))
        """
        if len(trajectory) < 3:
            return 0.0
        
        v1 = trajectory[1:-1] - trajectory[:-2]
        v2 = trajectory[2:] - trajectory[1:-1]
        
        # Normalize
        v1_norm = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-8)
        v2_norm = v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + 1e-8)
        
        cos_sim = np.sum(v1_norm * v2_norm, axis=1)
        curvature = np.mean(1 - cos_sim)
        
        return curvature
    
    def compute_heading(self, trajectory: np.ndarray) -> np.ndarray:
        """Compute heading angle at each point."""
        if len(trajectory) < 2:
            return np.array([0.0])
        
        # Heading from consecutive points
        diff = trajectory[1:] - trajectory[:-1]
        headings = np.arctan2(diff[:, 1], diff[:, 0])
        
        # Duplicate last heading
        headings = np.append(headings, headings[-1])
        
        return headings
    
    def compute_speed_profile(
        self,
        trajectory: np.ndarray,
        target_speed: float,
        max_accel: float = 2.0,
        max_decel: float = 3.0,
    ) -> np.ndarray:
        """
        Generate smooth speed profile along trajectory.
        
        Returns:
            speeds: Array of speeds at each trajectory point
        """
        n = len(trajectory)
        speeds = np.ones(n) * target_speed
        
        # Compute distances
        distances = np.zeros(n)
        for i in range(1, n):
            distances[i] = distances[i-1] + np.linalg.norm(trajectory[i] - trajectory[i-1])
        
        total_distance = distances[-1]
        if total_distance == 0:
            return speeds
        
        # Acceleration phase
        accel_dist = target_speed**2 / (2 * max_accel)
        # Deceleration phase
        decel_dist = target_speed**2 / (2 * max_decel)
        
        for i in range(n):
            d = distances[i]
            if d < accel_dist:
                # Accelerating
                speeds[i] = np.sqrt(2 * max_accel * d)
            elif d > total_distance - decel_dist:
                # Decelerating
                remaining = total_distance - d
                speeds[i] = np.sqrt(2 * max_decel * remaining)
        
        return np.clip(speeds, 0, target_speed)
