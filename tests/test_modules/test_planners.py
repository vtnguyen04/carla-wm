import pytest
import torch

from torch_wm.modules.planners.cem_planner import CEMPlanner
from torch_wm.modules.planners.gd_planner import GDPlanner
from torch_wm.modules.planners.hybrid_planner import HybridPlanner

def dummy_world_model(current_latent, action):
    B, H, A = action.shape
    B, D = current_latent.shape
    # Summarize action over horizon
    act_sum = action.sum(dim=(1, 2), keepdim=True)  # (B, 1, 1)
    final_latent = current_latent + act_sum.view(B, 1).expand(B, D)
    trajectory = action.unsqueeze(-1).expand(B, H, A, D).sum(dim=2)  # (B, H, D)
    return final_latent, trajectory

def setup_planner_env():
    B, D, A, H = 2, 4, 2, 5
    current_latent = torch.zeros(B, D)
    goal_latent = torch.ones(B, D)
    return current_latent, goal_latent, B, H, A

def test_cem_planner():
    planner = CEMPlanner(config={
        "samples": 10,
        "iterations": 2,
        "topk": 3,
        "action_dim": 2,
        "horizon": 5
    })
    planner.action_dim = 2
    planner.horizon = 5
    
    current_latent, goal_latent, B, H, A = setup_planner_env()
    actions = planner.plan(current_latent, dummy_world_model, goal_latent)
    assert actions.shape == (B, H, A)

def test_gd_planner():
    planner = GDPlanner(config={
        "iterations": 1,
        "lr": 0.1,
        "action_dim": 2,
        "horizon": 5
    })
    planner.action_dim = 2
    planner.horizon = 5
    
    current_latent, goal_latent, B, H, A = setup_planner_env()
    actions = planner.plan(current_latent, dummy_world_model, goal_latent)
    assert actions.shape == (B, H, A)

def test_hybrid_planner():
    planner = HybridPlanner(config={
        "samples": 10,
        "iterations": 1,
        "topk": 3,
        "lr": 0.1,
        "action_dim": 2,
        "horizon": 5
    })
    planner.action_dim = 2
    planner.horizon = 5
    
    current_latent, goal_latent, B, H, A = setup_planner_env()
    actions = planner.plan(current_latent, dummy_world_model, goal_latent)
    assert actions.shape == (B, H, A)
