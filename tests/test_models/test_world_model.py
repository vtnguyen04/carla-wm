import pytest
import torch
import numpy as np
import unittest.mock

from torch_wm.models.wm_agent import WMAgent
from torch_wm.utils.preprocessing import preprocess_obs

@pytest.fixture
def test_model():
    import os, ruamel.yaml as yaml
    from torch_wm.structs import AttrDict
    path = os.path.join(os.path.dirname(__file__), "../../torch_wm/rl/config/dreamerv3.yaml")
    model_configs = yaml.YAML(typ='safe').load(open(path))
    config = AttrDict(model_configs['defaults'])
    model = WMAgent(env_name="test_env", override_config=config, skip_env=True)
    model.compile()
    
    # We must patch the metrics/losses internally to avoid real WandB / full training loop dependencies
    model.world_model.loss_manager.active_losses = [] # disable all complex losses for unit test forward pass assertion
    
    return model

def test_world_model_forward(test_model, dummy_batch):
    B, L = 2, 4
    test_model.config.L = L  # Add config.L requirement for world_model indexing
    test_model.encoder_network.dim_concat = 256
    test_model.encoder_network.forward = unittest.mock.MagicMock()
    test_model.encoder_network.forward.return_value = {
        "stoch": torch.randn(B, L, test_model.config.stoch_size, test_model.config.discrete),
        "logits": torch.randn(B, L, test_model.config.stoch_size, test_model.config.discrete),
        "latent": torch.randn(B, L, 256),
        "state_seq": torch.randn(B, L, 256),
        "hidden": [(torch.randn(B, L, test_model.config.hidden_size), torch.randn(B, L, test_model.config.hidden_size))],
    }
    test_model.decoder_network.forward = unittest.mock.MagicMock(return_value={})
    test_model.reward_network.forward = unittest.mock.MagicMock(return_value=unittest.mock.MagicMock())
    test_model.continue_network.forward = unittest.mock.MagicMock(return_value=unittest.mock.MagicMock())
    test_model.encoder_network._tssm_branches = ["camera"]
    
    s = {"camera": torch.randint(0, 255, (B, L, 3, 64, 64), dtype=torch.uint8)}
    a = torch.randn(B, L, 15)
    r = torch.randn(B, L)
    d = torch.zeros(B, L, dtype=torch.bool)
    f = torch.zeros(B, L, dtype=torch.bool)
    f[:, 0] = True
    
    inputs = (s, a, r, d, f)
    total_loss = test_model.world_model(inputs)
    
    assert hasattr(test_model, "detached_posts")
    assert "hidden" in test_model.detached_posts
    assert "stoch" in test_model.detached_posts
    assert test_model.detached_posts["stoch"].dim() == 4
    assert getattr(test_model, "detached_is_firsts").shape == (B*L, 1)
