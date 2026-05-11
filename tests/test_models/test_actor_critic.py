import pytest
import torch

from torch_wm.models.wm_agent import WMAgent

@pytest.fixture
def test_model():
    import os, ruamel.yaml as yaml
    from torch_wm.structs import AttrDict
    path = os.path.join(os.path.dirname(__file__), "../../torch_wm/rl/config/dreamerv3.yaml")
    model_configs = yaml.YAML(typ='safe').load(open(path))
    config = AttrDict(model_configs['defaults'])
    model = WMAgent(env_name="test_env", override_config=config, skip_env=True)
    model.compile()
    
    # We must mock detached_posts from the world_model
    B, L = 2, 4
    
    # For actor model, detached_posts contains flattened sequences: (B*L, 1, D)
    N = B * L
    
    stoch_dim = model.config.stoch_size
    discrete = model.config.discrete
    rssm_dim = model.config.hidden_size
    rssm_layers = model.config.num_blocks_trans
    att_context_left = model.config.att_context_left
    H = model.config.get("H", 15)
    
    model.detached_posts = {
        "stoch": torch.randn(N, 1, stoch_dim, discrete),
        "logits": torch.randn(N, 1, stoch_dim, discrete),
        "deter": torch.randn(N, 1, model.config.hidden_size),
        "hidden": [(torch.randn(N, att_context_left, rssm_dim),
                    torch.randn(N, att_context_left, rssm_dim))] * rssm_layers
    }
    model.detached_is_firsts = torch.zeros(N, 1, dtype=torch.bool)
    model.detached_is_firsts_hidden = torch.zeros(N, att_context_left, dtype=torch.bool)
    
    # Needs some reward tracking for lambda returns
    import unittest.mock
    mock_dist = unittest.mock.MagicMock()
    mock_dist.mode = torch.randn(N, 1 + H, 1)
    
    model.reward_network.forward = unittest.mock.MagicMock(return_value=mock_dist)
    model.continue_network.forward = unittest.mock.MagicMock(return_value=mock_dist)
    
    mock_dist_v = unittest.mock.MagicMock()
    mock_dist_v.mode = torch.randn(N, 1 + H, 1)
    model.v_target.forward = unittest.mock.MagicMock(return_value=mock_dist_v)
    
    return model

def test_actor_critic_forward(test_model):
    
    # Create valid dummy tuple instead of Nones
    B, L = 2, 4
    num_actions = getattr(test_model.config, 'num_actions', 15)
    a = torch.randn(B, L, num_actions)
    r = torch.randn(B, L)
    d = torch.zeros(B, L, dtype=torch.bool)
    f = torch.zeros(B, L, dtype=torch.bool)
    f[:, 0] = True
    inputs = ({}, a, r, d, f)
    
    # Test ActorModel
    actor_loss = test_model.actor_model(inputs)
    
    # Should perform imagination and return loss
    assert isinstance(actor_loss, torch.Tensor)
    
    # Verify outputs populated for Critic
    assert hasattr(test_model, "detached_feats")
    assert hasattr(test_model, "detached_returns")
    
    # Test CriticModel
    critic_loss = test_model.critic_model(inputs)
    
    # Should use imagined trajectories and compute lambda returns
    assert isinstance(critic_loss, torch.Tensor)
    assert critic_loss.dim() == 0 # scalar loss
