"""
Stanley Controller for Trajectory Tracking

Industry-standard lateral control used by Stanford DARPA Grand Challenge team.
Combines heading error + cross-track error for smooth, accurate path tracking.

Control law:
    delta = -theta_e - atan2(k * e_y / v_x, 1)

where:
    theta_e = heading error (rad)
    e_y = lateral cross-track error (m)
    v_x = longitudinal velocity (m/s)
    k = cross-track gain (tunable)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

class StanleyController:
    """
    Stanley Controller for lateral control.
    
    Tracks a reference trajectory by minimizing:
    - Heading error (direction mismatch)
    - Cross-track error (lateral offset)
    """
    
    def __init__(
        self,
        wheelbase: float = 2.87,
        k_cross: float = 2.0,
        k_heading: float = 1.0,
        max_steer_rad: float = 0.6,  # ~35 degrees
        lookahead_steps: int = 10,
        dt: float = 0.1,
    ):
        """
        Args:
            wheelbase: Vehicle wheelbase in meters
            k_cross: Cross-track error gain (higher = more aggressive correction)
            k_heading: Heading error gain
            max_steer_rad: Maximum steering angle in radians
            lookahead_steps: How many waypoints ahead to track
            dt: Control timestep
        """
        self.wheelbase = wheelbase
        self.k_cross = k_cross
        self.k_heading = k_heading
        self.max_steer = max_steer_rad
        self.lookahead = lookahead_steps
        self.dt = dt
        
        # State
        self.prev_steer = 0.0
        self.steer_rate_limit = 0.5  # rad/s max steering rate
    
    def compute(
        self,
        current_state: Dict,
        trajectory: np.ndarray,
        target_speed: float = 8.0,
    ) -> Dict[str, float]:
        """
        Compute control to track trajectory.
        
        Args:
            current_state: {
                'x': float,          # Current x position (m)
                'y': float,          # Current y position (m)
                'yaw': float,        # Current heading (rad)
                'speed': float,      # Current speed (m/s)
            }
            trajectory: np.ndarray of shape (N, 2) or (N, 4)
                - If (N, 2): [(x, y), ...]
                - If (N, 4): [(x, y, yaw, speed), ...]
            target_speed: Desired speed (m/s) - used if not in trajectory
            
        Returns:
            control: {
                'steer': float,      # Normalized steering [-1, 1]
                'throttle': float,   # Throttle [0, 1]
                'brake': float,      # Brake [0, 1]
            }
        """
        x = current_state['x']
        y = current_state['y']
        yaw = current_state['yaw']
        speed = max(current_state['speed'], 0.1)  # Avoid div by zero
        
        # 1. Find closest point on trajectory
        closest_idx = self._find_closest_point(x, y, trajectory)
        
        # 2. Find lookahead point
        lookahead_idx = min(closest_idx + self.lookahead, len(trajectory) - 1)
        target_point = trajectory[lookahead_idx]
        
        # 3. Compute cross-track error (lateral distance to trajectory)
        cross_track_error, target_yaw = self._compute_cross_track_error(
            x, y, yaw, trajectory, closest_idx
        )
        
        # 4. Compute heading error
        heading_error = target_yaw - yaw
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        # 5. Stanley control law
        # delta = -theta_e - atan2(k * e_y / v, 1)
        steer_rad = (
            -self.k_heading * heading_error
            - np.arctan2(self.k_cross * cross_track_error / speed, 1)
        )
        
        # 6. Clamp steering
        steer_rad = np.clip(steer_rad, -self.max_steer, self.max_steer)
        
        # 7. Rate limiting (smooth steering)
        steer_rad = self._limit_steer_rate(steer_rad)
        
        # 8. Normalize to [-1, 1]
        steer_normalized = steer_rad / self.max_steer
        
        # 9. Longitudinal control (speed tracking)
        traj_speed = target_speed
        if trajectory.shape[-1] >= 4:
            traj_speed = trajectory[lookahead_idx, 3]
        
        throttle, brake = self._compute_longitudinal_control(speed, traj_speed)
        
        return {
            'steer': float(steer_normalized),
            'throttle': float(throttle),
            'brake': float(brake),
            'cross_track_error': float(cross_track_error),
            'heading_error': float(heading_error),
            'target_yaw': float(target_yaw),
        }
    
    def _find_closest_point(
        self, x: float, y: float, trajectory: np.ndarray
    ) -> int:
        """Find index of closest point on trajectory."""
        positions = trajectory[:, :2]  # (x, y)
        distances = np.sum((positions - np.array([x, y])) ** 2, axis=1)
        return int(np.argmin(distances))
    
    def _compute_cross_track_error(
        self,
        x: float,
        y: float,
        yaw: float,
        trajectory: np.ndarray,
        closest_idx: int,
    ) -> Tuple[float, float]:
        """
        Compute lateral cross-track error and target heading.
        
        Returns:
            cross_track_error: Lateral distance to trajectory (m), positive = left
            target_yaw: Heading of trajectory at closest point (rad)
        """
        # Get trajectory heading at closest point
        if trajectory.shape[-1] >= 4:
            # Trajectory has yaw information
            target_yaw = trajectory[closest_idx, 2]
        else:
            # Compute heading from trajectory geometry
            next_idx = min(closest_idx + 1, len(trajectory) - 1)
            dx = trajectory[next_idx, 0] - trajectory[closest_idx, 0]
            dy = trajectory[next_idx, 1] - trajectory[closest_idx, 1]
            target_yaw = np.arctan2(dy, dx)
        
        # Compute lateral error
        # Project vehicle position onto trajectory normal
        traj_heading_vec = np.array([np.cos(target_yaw), np.sin(target_yaw)])
        traj_normal = np.array([-traj_heading_vec[1], traj_heading_vec[0]])
        
        closest_pos = trajectory[closest_idx, :2]
        error_vec = np.array([x, y]) - closest_pos
        
        # Cross-track error (signed: positive = left of trajectory)
        cross_track_error = np.dot(error_vec, traj_normal)
        
        return cross_track_error, target_yaw
    
    def _limit_steer_rate(self, steer_rad: float) -> float:
        """Limit steering rate to prevent jerky steering."""
        max_delta = self.steer_rate_limit * self.dt
        steer_delta = steer_rad - self.prev_steer
        steer_delta = np.clip(steer_delta, -max_delta, max_delta)
        new_steer = self.prev_steer + steer_delta
        self.prev_steer = new_steer
        return new_steer
    
    def _compute_longitudinal_control(
        self, current_speed: float, target_speed: float
    ) -> Tuple[float, float]:
        """
        Simple PID speed controller.
        
        Returns:
            throttle: [0, 1]
            brake: [0, 1]
        """
        speed_error = target_speed - current_speed
        
        # P controller for throttle
        if speed_error > 0:
            throttle = np.clip(0.5 * speed_error, 0.0, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.clip(-0.8 * speed_error, 0.0, 1.0)
        
        return throttle, brake
    
    def reset(self):
        """Reset controller state."""
        self.prev_steer = 0.0
