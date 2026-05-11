import pytest
import torch
from torch_wm.structs import AttrDict
from torch_wm.modules.networks.encoder_network import MultiEncoderNetwork

class SpaceStub:
    def __init__(self, shape):
        self.shape = shape

@pytest.fixture
def dummy_obs_space():
    return {
        "camera": SpaceStub((3, 64, 64)),
        "vector": SpaceStub((10,)),
        "lidar": SpaceStub((1, 64, 64)),  # another image but signal-only
        "reward": SpaceStub(())
    }

@pytest.fixture
def dummy_obs_config():
    return {
        "enabled": ["camera", "vector", "lidar"],
        "camera": {"tssm": True},
        "vector": {"tssm": True},
        # lidar has tssm=False by default if not specified or specified False
        "lidar": {"tssm": False}
    }

def test_multi_encoder_init(dummy_obs_space, dummy_obs_config):
    # Test initialization sets up branches correctly
    encoder = MultiEncoderNetwork(
        obs_space=dummy_obs_space,
        obs_config=dummy_obs_config,
        dim_cnn=32,
        stoch_size=16,
        discrete=16
    )
    
    assert len(encoder.encoders) == 2 # camera, lidar are conv. vector is skipped (dim < 3)
    
    assert "camera" in encoder._tssm_branches
    assert "vector" not in encoder._tssm_branches
    assert "lidar" in encoder._signal_branches
    
    assert encoder.dim_concat > 0

def test_multi_encoder_forward(dummy_obs_space, dummy_obs_config):
    encoder = MultiEncoderNetwork(
        obs_space=dummy_obs_space,
        obs_config=dummy_obs_config,
        dim_cnn=32,
        stoch_size=16,
        discrete=16
    )
    
    # B=2 single step forward
    inputs = {
        "camera": torch.rand(2, 3, 64, 64),
        "vector": torch.rand(2, 10),
        "lidar": torch.rand(2, 1, 64, 64)
    }
    
    out = encoder(inputs)
    
    assert "stoch" in out
    assert "latent" in out
    assert "logits" in out
    assert "signal" in out
    
    assert out["stoch"].shape == (2, 16, 16) # B, stoch_size, discrete
    assert out["logits"].shape == (2, 16, 16)
    assert out["latent"].shape == (2, encoder.dim_concat)
    assert out["signal"].shape == (2, encoder.dim_signal)

def test_multi_encoder_forward_seq(dummy_obs_space, dummy_obs_config):
    encoder = MultiEncoderNetwork(
        obs_space=dummy_obs_space,
        obs_config=dummy_obs_config,
        dim_cnn=32,
        stoch_size=16,
        discrete=16
    )
    
    # B=2, T=3 sequence forward
    inputs = {
        "camera": torch.rand(2, 3, 3, 64, 64),
        "vector": torch.rand(2, 3, 10),
        "lidar": torch.rand(2, 3, 1, 64, 64)
    }
    
    out = encoder(inputs)
    
    assert out["stoch"].shape == (2, 3, 16, 16)
    assert out["logits"].shape == (2, 3, 16, 16)
    assert out["latent"].shape == (2, 3, encoder.dim_concat)
    assert out["signal"].shape == (2, 3, encoder.dim_signal)

def test_multi_encoder_missing_tssm_inputs(dummy_obs_space, dummy_obs_config):
    encoder = MultiEncoderNetwork(
        obs_space=dummy_obs_space,
        obs_config=dummy_obs_config,
        dim_cnn=32,
        stoch_size=16,
        discrete=16
    )
    
    # Missing all TSSM inputs
    inputs = {
        "lidar": torch.rand(2, 1, 64, 64)
    }
    
    with pytest.raises(ValueError, match="No TSSM sensor data found"):
        encoder(inputs)
