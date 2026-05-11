"""Shared pytest fixtures for the WMAgent test suite."""

import pytest
import torch
import numpy as np

from torch_wm.structs import AttrDict

@pytest.fixture
def dummy_config():
    """Minimal WMAgent configuration needed for instantiation."""
    config = AttrDict()
    config.stoch_size = 32
    config.discrete = 32
    config.hidden_size = 256
    
    # Encoder
    config.base_dim = 16
    config.cnn_depth = 2
    
    # Decoder
    config.decoder_base_dim = 16
    config.decoder_cnn_depth = 2
    
    # TSSM
    config.rssm_type = "tssm"
    config.rssm_dim = 128
    config.rssm_layers = 1
    config.rssm_heads = 4
    config.L = 4
    config.att_context_left = 4
    
    # Network layers
    config.num_layers = 1
    config.layer_norm = True
    config.precision = "float32"
    
    config.env_name = "dummy_env"
    return config

@pytest.fixture
def dummy_batch():
    """Simulates a batch of data exactly as it comes from embodied (HWC)."""
    B, L = 2, 4
    batch = {
        "camera": np.random.randint(0, 255, (B, L, 64, 64, 3), dtype=np.uint8),
        "reward": np.random.randn(B, L).astype(np.float32),
        "is_terminal": np.zeros((B, L), dtype=bool),
        "is_first": np.zeros((B, L), dtype=bool),
        "action": np.random.randn(B, L, 2).astype(np.float32),
    }
    # Set boundaries
    batch["is_first"][:, 0] = True
    return batch
