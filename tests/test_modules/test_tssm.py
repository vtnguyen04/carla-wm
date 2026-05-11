import pytest
import torch
from torch_wm.modules.dynamics.tssm import TSSM

@pytest.fixture
def tssm():
    return TSSM(
        stoch_size=32,
        discrete=32,
        hidden_size=256,
        num_actions=15
    )

def test_tssm_initial(tssm):
    batch_size = 4
    seq_length = 5
    init_state = tssm.initial(batch_size, seq_length)
    
    assert "stoch" in init_state
    assert "deter" in init_state
    assert "logits" in init_state
    
    assert init_state["stoch"].shape == (batch_size, seq_length, 32, 32)
    assert init_state["deter"].shape == (batch_size, seq_length, 256)
    assert init_state["logits"].shape == (batch_size, seq_length, 32, 32)

def test_tssm_observe(tssm):
    B, T = 2, 3
    # embed from multi encoder: stoch and logits
    embed = {
        "stoch": torch.randn(B, T, 32, 32),
        "logits": torch.randn(B, T, 32, 32)
    }
    action = torch.randn(B, T, 15)
    is_first = torch.zeros(B, T)
    is_first[0, 0] = 1.0 # Simulate one sequence starting
    
    post, prior = tssm.observe(embed, action, is_first)
    
    assert post["stoch"].shape == (B, T, 32, 32)
    assert post["deter"].shape == (B, T, 256)
    assert prior["stoch"].shape == (B, T, 32, 32)

def test_tssm_imagine(tssm):
    B = 2
    # Imagine 4 steps into the future
    action = torch.randn(B, 4, 15)
    
    # Start from an initial state B, 1
    state = tssm.initial(B, 1)
    
    # Mock policy net
    class MockPolicy:
        def __init__(self):
            pass
        def rsample(self):
            return torch.randn(B, 1, 15)
            
    p_net = lambda feat: MockPolicy()
    
    prior = tssm.imagine(p_net=p_net, prev_state=state, img_steps=2, actions=action)
    
    assert prior["stoch"].shape == (B, 3, 32, 32) # includes initial state, so 1+2 = 3 steps
    assert prior["deter"].shape == (B, 3, 256)

def test_tssm_get_feat(tssm):
    B, L = 2, 3
    state = {
        "stoch": torch.randn(B, L, 32, 32),
        "deter": torch.randn(B, L, 256)
    }
    
    feat = tssm.get_feat(state)
    
    # Flattened stoch (32*32=1024) + deter (256) = 1280
    assert feat.shape == (B, L, 1280)
    
    # Test single-step (no L dim)
    state_single = {
        "stoch": torch.randn(B, 32, 32),
        "deter": torch.randn(B, 256)
    }
    feat_single = tssm.get_feat(state_single)
    assert feat_single.shape == (B, 1280)

def test_tssm_get_dist(tssm):
    B, L = 2, 3
    state = {
        "logits": torch.randn(B, L, 32, 32)
    }
    dist = tssm.get_dist(state)
    
    assert hasattr(dist, "sample")
    sample = dist.sample()
    assert sample.shape == (B, L, 32, 32)
