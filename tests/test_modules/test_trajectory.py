import pytest
import torch
import numpy as np
from torch_wm.modules.trajectory.bezier import BezierTrajectoryGenerator
from torch_wm.modules.rewards.trajectory_reward import TrajectoryReward
from torch_wm.modules.actors.bezier_actor import BezierActor

def test_bezier_numpy():
    generator = BezierTrajectoryGenerator(num_samples=10)
    
    # Single numpy
    points = np.array([[0, 0], [1, 1], [2, -1], [3, 0]])
    traj = generator.generate(points)
    assert traj.shape == (10, 2)
    assert np.allclose(traj[0], [0, 0])
    assert np.allclose(traj[-1], [3, 0])
    
    # Batched numpy
    points_batched = np.stack([points, points])
    traj_batched = generator.generate(points_batched)
    assert traj_batched.shape == (2, 10, 2)
    
def test_bezier_torch():
    generator = BezierTrajectoryGenerator(num_samples=10)
    
    # Single torch
    points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, -1.0], [3.0, 0.0]])
    traj = generator.generate(points)
    assert traj.shape == (10, 2)
    assert torch.allclose(traj[0], torch.tensor([0.0, 0.0]))
    assert torch.allclose(traj[-1], torch.tensor([3.0, 0.0]))
    
    # Batched torch
    points_batched = torch.stack([points, points])
    traj_batched = generator.generate(points_batched)
    assert traj_batched.shape == (2, 10, 2)

def test_bezier_functions():
    generator = BezierTrajectoryGenerator(num_samples=10)
    points = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
    traj = generator.generate(points)
    
    curv = generator.compute_curvature(traj)
    assert isinstance(curv, float)
    
    heading = generator.compute_heading(traj)
    assert heading.shape == (10,)
    
    speed = generator.compute_speed_profile(traj, target_speed=5.0)
    assert speed.shape == (10,)

def test_trajectory_reward():
    reward = TrajectoryReward()
    trajectory = torch.randn(2, 10, 2)
    lane_centerline = torch.randn(5, 2)
    obstacles = torch.randn(3, 2)
    current_state = {'x': torch.zeros(2), 'y': torch.zeros(2), 'yaw': torch.zeros(2), 'speed': torch.zeros(2)}
    
    r = reward.forward(trajectory, lane_centerline, current_state, obstacles)
    assert r.shape == (2,)

def test_bezier_actor():
    actor = BezierActor(feat_size=128)
    feat = torch.randn(2, 128)
    
    out = actor.forward(feat)
    assert 'control_points' in out
    assert 'speeds' in out
    
    traj = actor.get_trajectory(feat)
    assert 'trajectory' in traj
    assert 'control_points' in traj
    assert 'speeds' in traj
