"""
Trajectory Reward Module

Computes reward based on trajectory quality:
- Lane tracking: How well trajectory follows lane centerline
- Smoothness: Curvature of trajectory
- Progress: Forward movement
- Collision avoidance: Distance to obstacles
- Speed consistency: Smooth speed profile
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

class TrajectoryReward(nn.Module):
    """
    Computes reward for predicted trajectories.
    Used during RL training to guide actor learning.
    """
    
    def __init__(
        self,
        weights: Optional[dict] = None,
    ):
        """
        Args:
            weights: Dict of reward component weights
        """
        super().__init__()
        
        self.weights = weights or {
            'lane_tracking': 10.0,
            'smoothness': 1.0,
            'progress': 2.0,
            'speed_tracking': 1.0,
            'collision_avoidance': 50.0,
            'heading_alignment': 2.0,
        }
    
    def forward(
        self,
        trajectory: torch.Tensor,
        lane_centerline: torch.Tensor,
        current_state: dict,
        obstacles: Optional[torch.Tensor] = None,
        target_speed: float = 8.0,
        horizon_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute trajectory reward.
        
        Args:
            trajectory: [B, N, 2] - Predicted trajectory (x, y)
            lane_centerline: [M, 2] - Lane centerline waypoints
            current_state: {
                'x': [B], 'y': [B], 'yaw': [B], 'speed': [B]
            }
            obstacles: [K, 2] - Obstacle positions (optional)
            target_speed: Desired speed (m/s)
            horizon_weights: [N] - Weights for each timestep (optional)
        
        Returns:
            reward: [B] - Total reward per sample
        """
        B, N, _ = trajectory.shape
        total_reward = torch.zeros(B, device=trajectory.device)
        
        # 1. Lane tracking reward
        lane_reward = self._compute_lane_tracking_reward(
            trajectory, lane_centerline
        )
        total_reward += self.weights['lane_tracking'] * lane_reward
        
        # 2. Smoothness reward (curvature)
        smoothness_reward = self._compute_smoothness_reward(trajectory)
        total_reward += self.weights['smoothness'] * smoothness_reward
        
        # 3. Progress reward
        progress_reward = self._compute_progress_reward(
            trajectory, current_state
        )
        total_reward += self.weights['progress'] * progress_reward
        
        # 4. Speed tracking reward
        speed_reward = self._compute_speed_reward(
            trajectory, current_state, target_speed
        )
        total_reward += self.weights['speed_tracking'] * speed_reward
        
        # 5. Collision avoidance reward
        if obstacles is not None:
            collision_reward = self._compute_collision_reward(
                trajectory, obstacles
            )
            total_reward += self.weights['collision_avoidance'] * collision_reward
        
        # 6. Heading alignment reward
        heading_reward = self._compute_heading_alignment_reward(
            trajectory, lane_centerline, current_state
        )
        total_reward += self.weights['heading_alignment'] * heading_reward
        
        return total_reward
    
    def _compute_lane_tracking_reward(
        self,
        trajectory: torch.Tensor,
        lane_centerline: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reward based on how close trajectory stays to lane centerline.
        Negative MSE distance to nearest centerline point.
        """
        B, N, _ = trajectory.shape
        M, _ = lane_centerline.shape
        
        # Expand for broadcasting: trajectory [B, N, 1, 2], centerline [1, 1, M, 2]
        traj_exp = trajectory.unsqueeze(2)  # [B, N, 1, 2]
        center_exp = lane_centerline.unsqueeze(0).unsqueeze(0)  # [1, 1, M, 2]
        
        # Compute distances: [B, N, M]
        distances = torch.sum((traj_exp - center_exp) ** 2, dim=-1)
        
        # Find minimum distance for each trajectory point: [B, N]
        min_distances = torch.min(distances, dim=-1).values  # [B, N]
        
        # Reward: negative mean squared distance
        reward = -torch.mean(min_distances, dim=-1)  # [B]
        
        return reward
    
    def _compute_smoothness_reward(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reward for smooth trajectory (low curvature).
        curvature = 1 - cos(angle between consecutive segments)
        """
        if trajectory.shape[1] < 3:
            return torch.zeros(trajectory.shape[0], device=trajectory.device)
        
        # Compute velocity vectors
        v1 = trajectory[:, 1:-1] - trajectory[:, :-2]  # [B, N-2, 2]
        v2 = trajectory[:, 2:] - trajectory[:, 1:-1]   # [B, N-2, 2]
        
        # Normalize
        v1_norm = v1 / (torch.norm(v1, dim=-1, keepdim=True) + 1e-8)
        v2_norm = v2 / (torch.norm(v2, dim=-1, keepdim=True) + 1e-8)
        
        # Cosine similarity
        cos_sim = torch.sum(v1_norm * v2_norm, dim=-1)  # [B, N-2]
        
        # Curvature: 0 = straight, 2 = sharp turn
        curvature = 1 - cos_sim
        
        # Reward: negative mean curvature
        reward = -torch.mean(curvature, dim=-1)  # [B]
        
        return reward
    
    def _compute_progress_reward(
        self,
        trajectory: torch.Tensor,
        current_state: dict,
    ) -> torch.Tensor:
        """
        Reward for moving forward along x-axis.
        """
        start_x = trajectory[:, 0, 0]
        end_x = trajectory[:, -1, 0]
        
        progress = end_x - start_x  # [B]
        
        return progress
    
    def _compute_speed_reward(
        self,
        trajectory: torch.Tensor,
        current_state: dict,
        target_speed: float,
    ) -> torch.Tensor:
        """
        Reward for maintaining target speed.
        Approximate speed from trajectory displacement.
        """
        # Compute displacement per step
        displacement = trajectory[:, 1:] - trajectory[:, :-1]  # [B, N-1, 2]
        step_distances = torch.norm(displacement, dim=-1)  # [B, N-1]
        
        # Approximate speed (assuming 0.1s per step)
        speeds = step_distances / 0.1  # [B, N-1]
        
        # Reward: negative deviation from target
        speed_error = (speeds - target_speed) ** 2
        reward = -torch.mean(speed_error, dim=-1)  # [B]
        
        return reward
    
    def _compute_collision_reward(
        self,
        trajectory: torch.Tensor,
        obstacles: torch.Tensor,
        safety_distance: float = 3.0,
    ) -> torch.Tensor:
        """
        Heavy penalty for getting too close to obstacles.
        """
        B, N, _ = trajectory.shape
        K, _ = obstacles.shape
        
        # Expand: trajectory [B, N, 1, 2], obstacles [1, 1, K, 2]
        traj_exp = trajectory.unsqueeze(2)
        obs_exp = obstacles.unsqueeze(0).unsqueeze(0)
        
        # Distances: [B, N, K]
        distances = torch.sqrt(torch.sum((traj_exp - obs_exp) ** 2, dim=-1) + 1e-8)
        
        # Minimum distance to any obstacle
        min_dist = torch.min(distances, dim=-1).values  # [B, N]
        
        # Penalty if closer than safety distance
        violation = F.relu(safety_distance - min_dist)  # [B, N]
        penalty = torch.mean(violation ** 2, dim=-1)  # [B]
        
        return -penalty
    
    def _compute_heading_alignment_reward(
        self,
        trajectory: torch.Tensor,
        lane_centerline: torch.Tensor,
        current_state: dict,
    ) -> torch.Tensor:
        """
        Reward for trajectory heading matching lane direction.
        """
        if trajectory.shape[1] < 2:
            return torch.zeros(trajectory.shape[0], device=trajectory.device)
        
        # Trajectory heading
        traj_diff = trajectory[:, 1:] - trajectory[:, :-1]
        traj_heading = torch.atan2(traj_diff[:, :, 1], traj_diff[:, :, 0])  # [B, N-1]
        
        # Find nearest centerline heading for each trajectory point
        B, N_minus_1, _ = traj_diff.shape
        M, _ = lane_centerline.shape
        
        traj_points = trajectory[:, :-1].unsqueeze(2)  # [B, N-1, 1, 2]
        center_points = lane_centerline.unsqueeze(0).unsqueeze(0)  # [1, 1, M, 2]
        
        distances = torch.sum((traj_points - center_points) ** 2, dim=-1)  # [B, N-1, M]
        nearest_idx = torch.argmin(distances, dim=-1)  # [B, N-1]
        
        # Compute centerline heading at nearest points
        centerline_diff = lane_centerline[1:] - lane_centerline[:-1]
        centerline_heading = torch.atan2(centerline_diff[:, 1], centerline_diff[:, 0])  # [M-1]
        
        # Simple approximation: use average centerline heading
        avg_centerline_heading = torch.atan2(
            lane_centerline[-1, 1] - lane_centerline[0, 1],
            lane_centerline[-1, 0] - lane_centerline[0, 0]
        )
        
        # Heading alignment
        heading_diff = traj_heading - avg_centerline_heading
        heading_diff = torch.atan2(torch.sin(heading_diff), torch.cos(heading_diff))
        
        # Reward: cosine alignment
        reward = torch.mean(torch.cos(heading_diff), dim=-1)  # [B]
        
        return reward
