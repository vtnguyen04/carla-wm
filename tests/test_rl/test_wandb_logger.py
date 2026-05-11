import pytest
import torch
import numpy as np
from unittest.mock import MagicMock
from torch_wm.rl.loggers.wandb_logger import WandBLogger

def test_wandb_logger_full_coverage(monkeypatch):
    # Mock wandb module
    mock_wandb = MagicMock()
    mock_wandb.Settings = MagicMock()
    mock_wandb.Video = lambda x, **kw: "mock_video"
    mock_wandb.Image = lambda x, **kw: "mock_image"
    mock_wandb.Histogram = lambda x, **kw: "mock_hist"
    mock_wandb.Table = lambda **kw: "mock_table"
    mock_wandb.plot = MagicMock()
    
    # Mock actual wandb import
    import sys
    sys.modules["wandb"] = mock_wandb

    logger = WandBLogger(config={"test": True}, log_dir=".", project="test")
    assert logger.enabled == True
    
    # Test Scalars
    logger.log_step({"loss_actor": 1.0, "cpc_loss": 0.5, "speed": 10.0}, 1)
    logger.log_epoch({"loss_critic": 1.0, "wm_kl": 0.1, "actor_ent": 0.2, "critic_val": 0.3, "other": 0.0}, 1, 1)

    # Test Action Distribution
    batch = {"action": torch.randint(0, 15, (2, 10, 15))}
    logger.log_action_distribution(batch, 1)
    
    # Test Reward Analysis
    model_outputs = {
        "model_rewards": MagicMock(mode=lambda: torch.ones(2, 10))
    }
    batch["reward"] = torch.ones(2, 10)
    logger.log_reward_analysis(batch, model_outputs, 1)
    
    # Test Latent Stats
    model_outputs = {
        "posts": {
            "logits": torch.randn(2, 10, 32),
            "stoch": torch.randn(2, 10, 32)
        },
        "priors": {
            "logits": torch.randn(2, 10, 32)
        }
    }
    logger.log_latent_stats(model_outputs, 1)
    
    # Test Gradient Norms
    mock_model = MagicMock()
    mock_param = MagicMock(grad=MagicMock(data=torch.ones(10)))
    mock_module = MagicMock()
    mock_module.parameters.return_value = [mock_param]
    
    mock_model.encoder_network = mock_module
    mock_model.decoder_network = mock_module
    mock_model.dynamics_model = mock_module
    mock_model.policy_network = mock_module
    mock_model.value_network = mock_module
    mock_model.reward_network = mock_module
    
    logger.log_gradient_norms(mock_model, 1)
    
    # Test Reconstructions
    mock_rec_dist = MagicMock(mode=lambda: torch.zeros(2, 10, 3, 64, 64))
    mock_model._last_outputs = {
        "states_rec_dist": {"camera": mock_rec_dist}
    }
    batch["camera"] = torch.zeros(2, 10, 3, 64, 64)
    logger.log_reconstructions(mock_model, batch, 1)
    
    # Finish
    logger.finish()
