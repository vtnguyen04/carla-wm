import pytest
import torch
from torch_wm.modules.dynamics.rssm import RSSM

@pytest.fixture
def rssm():
    return RSSM(
        stoch_size=32,
        discrete=32,
        hidden_size=256,
        num_actions=15
    )

def test_rssm_initial(rssm):
    batch_size = 4
    seq_length = 2
    
    init_state = rssm.initial(batch_size, seq_length)
    
    assert init_state["deter"].shape == (batch_size, seq_length, 256)
    assert init_state["stoch"].shape == (batch_size, seq_length, 32, 32)
    assert init_state["logits"].shape == (batch_size, seq_length, 32, 32)
    assert "hidden" in init_state

def test_rssm_observe(rssm):
    batch_size = 4
    seq_length = 5
    embed_size = 32 * 32
    
    embed = torch.randn(batch_size, seq_length, embed_size)
    action = torch.randn(batch_size, seq_length, 15)
    is_first = torch.zeros(batch_size, seq_length)
    is_first[:, 0] = 1.0  # First step is true
    
    states = {"stoch": embed.view(batch_size, seq_length, 32, 32)}
    
    post, prior = rssm.observe(states, action, is_first)
    
    assert post["deter"].shape == (batch_size, seq_length, 256)
    assert prior["deter"].shape == (batch_size, seq_length, 256)
    assert post["stoch"].shape == (batch_size, seq_length, 32, 32)
    assert prior["stoch"].shape == (batch_size, seq_length, 32, 32)

def test_rssm_imagine(rssm):
    B = 2
    
    state = rssm.initial(B, 1)
    
    # Dummy policy network
    class DummyPolicy:
        def __call__(self, feat):
            class DummyDist:
                def rsample(self):
                    return torch.randn(B, 15)
            return DummyDist()
            
    p_net = DummyPolicy()
    
    img_states = rssm.imagine(p_net=p_net, prev_state=state, img_steps=3)
    
    # 3 steps imagined + 1 initial = 4 steps
    # Wait, the imagine loop creates img_steps steps, plus the initial state.
    # Total seq length should be 4
    assert img_states["deter"].shape == (B, 4, 256)
    assert img_states["stoch"].shape == (B, 4, 32, 32)
    assert "action" in img_states
    
def test_rssm_get_feat(rssm):
    state = rssm.initial(2, 5)
    
    feat = rssm.get_feat(state)
    
    # stoch is 32x32 = 1024
    # deter is 256
    # total = 1280
    assert feat.shape == (2, 5, 1280)
