import pytest
import torch
from torch_wm.modules.networks.policy_network import PolicyNetwork
from torch_wm.distributions import OneHotDist

def test_policy_network_discrete():
    policy = PolicyNetwork(
        num_actions=15,
        hidden_size=64,
        num_mlp_layers=2,
        feat_size=128,
        discrete=True
    )
    
    B, T = 2, 3
    x = torch.randn(B, T, 128)
    
    dist = policy(x)
    
    assert isinstance(dist, OneHotDist)
    sample = dist.sample()
    assert sample.shape == (B, T, 15)

def test_policy_network_continuous():
    policy = PolicyNetwork(
        num_actions=5,
        hidden_size=64,
        num_mlp_layers=2,
        feat_size=128,
        discrete=False,
        min_std=0.1,
        max_std=1.0
    )
    
    B = 2
    x = torch.randn(B, 128)
    
    dist = policy(x)
    
    # Independent(Normal)
    assert hasattr(dist, "base_dist")
    sample = dist.sample()
    assert sample.shape == (B, 5)
    
    # Check bounds on std
    std = dist.base_dist.scale
    assert torch.all(std >= 0.1)
    assert torch.all(std <= 1.0)
