"""
Pure Pursuit Controller

Simple geometric path tracking.
Follows a lookahead point on the reference trajectory.

Control law:
    delta = atan2(2 * L * sin(alpha) / l_d, 1)

where:
    L = wheelbase
    alpha = angle to lookahead point
    l_d = lookahead distance
"""

import numpy as np
from typing import Dict

class PurePursuitController:
    """
    Pure Pursuit Controller for lateral control.
    Simpler than Stanley but less accurate at high speeds.
    """
    
    def __init__(
        self,
        wheelbase: float = 2.87,
        lookahead_distance: float = 5.0,
        max_steer_rad: float = 0.6,
        k_speed: float = 0.5,
    ):
        """
        Args:
            wheelbase: Vehicle wheelbase (m)
            lookahead_distance: Base lookahead distance (m)
            max_steer_rad: Maximum steering angle (rad)
            k_speed: Speed-dependent lookahead gain
        """
        self.wheelbase = wheelbase
        self.base_lookahead = lookahead_distance
        self.max_steer = max_steer_rad
        self.k_speed = k_speed
        self.prev_steer = 0.0
        self.steer_rate_limit = 0.5
    
    def compute(
        self,
        current_state: Dict,
        trajectory: np.ndarray,
        target_speed: float = 8.0,
    ) -> Dict[str, float]:
        """
        Compute control to track trajectory.
        
        Args:
            current_state: {'x', 'y', 'yaw', 'speed'}
            trajectory: [(x, y) or (x, y, yaw, speed)]
            target_speed: Desired speed (m/s)
        
        Returns:
            {'steer', 'throttle', 'brake'}
        """
        x = current_state['x']
        y = current_state['y']
        yaw = current_state['yaw']
        speed = max(current_state['speed'], 0.1)
        
        # Speed-dependent lookahead
        lookahead = self.base_lookahead + self.k_speed * speed
        
        # Find lookahead point
        target_idx = self._find_lookahead_point(x, y, trajectory, lookahead)
        target = trajectory[target_idx, :2]
        
        # Angle to target
        alpha = np.arctan2(target[1] - y, target[0] - x) - yaw
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
        
        # Pure Pursuit law
        steer = np.arctan2(2 * self.wheelbase * np.sin(alpha) / lookahead, 1)
        steer = np.clip(steer, -self.max_steer, self.max_steer)
        
        # Rate limiting
        steer = self._limit_rate(steer)
        
        # Normalize
        steer_normalized = steer / self.max_steer
        
        # Longitudinal control
        throttle, brake = self._compute_longitudinal(speed, target_speed)
        
        return {
            'steer': float(steer_normalized),
            'throttle': float(throttle),
            'brake': float(brake),
        }
    
    def _find_lookahead_point(self, x, y, trajectory, lookahead):
        """Find point approximately lookahead distance ahead."""
        closest_idx = self._closest_point(x, y, trajectory)
        
        for i in range(closest_idx, len(trajectory)):
            dist = np.linalg.norm(trajectory[i, :2] - np.array([x, y]))
            if dist >= lookahead:
                return i
        
        return len(trajectory) - 1
    
    def _closest_point(self, x, y, trajectory):
        """Find closest trajectory point."""
        distances = np.sum((trajectory[:, :2] - np.array([x, y])) ** 2, axis=1)
        return int(np.argmin(distances))
    
    def _limit_rate(self, steer):
        """Limit steering rate."""
        max_delta = self.steer_rate_limit * 0.1
        delta = steer - self.prev_steer
        delta = np.clip(delta, -max_delta, max_delta)
        new_steer = self.prev_steer + delta
        self.prev_steer = new_steer
        return new_steer
    
    def _compute_longitudinal(self, current_speed, target_speed):
        """Speed control."""
        error = target_speed - current_speed
        if error > 0:
            return np.clip(0.5 * error, 0, 1), 0.0
        else:
            return 0.0, np.clip(-0.8 * error, 0, 1)
    
    def reset(self):
        self.prev_steer = 0.0
