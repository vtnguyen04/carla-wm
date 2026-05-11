import pytest
import torch
import os
import ruamel.yaml as yaml
from torch_wm.models.wm_agent import WMAgent
from torch_wm.structs import AttrDict
from torch_wm.core.registry import ModuleRegistry

# Dummy Observation Space
OBS_SPACE = {
    "camera": AttrDict({"shape": (3, 64, 64), "tssm": True}),
    "reward": AttrDict({"shape": (), "tssm": False}),
    "is_first": AttrDict({"shape": (), "tssm": False}),
    "is_last": AttrDict({"shape": (), "tssm": False}),
    "is_terminal": AttrDict({"shape": (), "tssm": False}),
}

ACT_SPACE = AttrDict({"shape": (15,), "discrete": True})

def load_config(name):
    path = f"torch_wm/rl/config/{name}.yaml"
    with open(path, "r") as f:
        cfg = yaml.YAML(typ='safe').load(f)
    return AttrDict(cfg['defaults'])

@pytest.mark.parametrize("algo_name", ["dreamerv3", "twister", "vjepa"])
def test_algorithm_pipeline(algo_name):
    print(f"\n[Testing Algorithm: {algo_name.upper()}]")
    config = load_config(algo_name)
    
    # Overrides for testing efficiency
    config.batch_size = 2
    config.batch_length = 4
    config.checkpoint_path = "" # Skip weight loading
    
    # All algorithms use standard 64x64 for testing efficiency
    obs_space = OBS_SPACE
    if algo_name == "vjepa":
        config.image_size = (64, 64)

    # 1. Test Loading/Initialization
    agent = WMAgent(
        env_name="test_env", 
        override_config=config, 
        skip_env=True, 
        obs_space=obs_space
    )
    agent.compile()
    
    print(f"  - Model loaded: {agent.dynamics_model.__class__.__name__}")
    
    # 2. Test Forward Pass (Encoding -> Dynamics)
    B, T = config.batch_size, config.batch_length
    obs = {
        "camera": torch.randn(B, T, *obs_space["camera"].shape),
    }
    action = torch.randn(B, T, 15)
    reward = torch.randn(B, T)
    continue_ = torch.ones(B, T)
    is_first = torch.zeros(B, T)
    is_first[:, 0] = 1.0
    
    # Test World Model Forward (train mode)
    agent.world_model.train()
    
    # WorldModel.forward expects (s, a, r, d, f)
    inputs = (obs, action, reward, continue_, is_first)
    loss = agent.world_model(inputs)
    
    # After forward, agent._last_outputs should be populated
    outputs = agent._last_outputs
    
    assert "posts" in outputs
    assert "priors" in outputs
    print("  - Forward pass success")
    
    assert not torch.isnan(loss), "World Model total loss is NaN"
    print(f"  - Forward pass & Loss computation success: {loss.item():.4f}")
    
    # Check metrics (WorldModel.forward adds them to itself)
    metrics = agent.world_model.added_metrics
    assert len(metrics) > 0, "No metrics generated in World Model"
    print(f"  - Metrics generated: {list(metrics.keys())}")

    # 4. Test Replay Buffer Interaction (Logic check)
    # Replay buffer is usually external to WMAgent, but let's check if we can 
    # simulate the data it would provide.
    # The compute_loss already tested the data format requirement.

def test_vjepa_specific_layers():
    """Detailed check for V-JEPA specific layers if loaded."""
    config = load_config("vjepa")
    config.checkpoint_path = ""
    config.image_size = (64, 64)
    obs_space = {"camera": AttrDict({"shape": (3, 64, 64), "tssm": True})}
    
    agent = WMAgent(env_name="test_vjepa", override_config=config, skip_env=True, obs_space=obs_space)
    
    # Check Encoder
    encoder = agent.encoder_network
    assert hasattr(encoder, "model"), "VJEPA Encoder should have a model (ViT)"
    
    # Check Predictor
    predictor = agent.dynamics_model
    assert hasattr(predictor, "predictor"), "VJEPA Predictor should have a internal predictor"
    
    print("  - V-JEPA specific layers verified")

if __name__ == "__main__":
    # Manual run if needed
    for algo in ["dreamerv3", "twister", "vjepa"]:
        test_algorithm_pipeline(algo)
