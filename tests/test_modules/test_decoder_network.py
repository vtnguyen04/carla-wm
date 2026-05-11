import pytest
import torch
from torch_wm.modules.networks.decoder_network import MultiDecoderNetwork

class SpaceStub:
    def __init__(self, shape):
        self.shape = shape

@pytest.fixture
def dummy_obs_space():
    return {
        "camera": SpaceStub((3, 64, 64)),
        "vector": SpaceStub((10,)),
        "lidar": SpaceStub((1, 64, 64)),  
        "nocamera": SpaceStub((3, 64, 64))
    }

@pytest.fixture
def dummy_obs_config():
    return {
        "enabled": ["camera", "vector", "lidar", "nocamera"],
        "camera": {"decode": True},
        "vector": {"decode": True},
        "lidar": {"decode": True},
        "nocamera": {"decode": False}
    }

def test_multi_decoder_init(dummy_obs_space, dummy_obs_config):
    decoder = MultiDecoderNetwork(
        obs_space=dummy_obs_space,
        obs_config=dummy_obs_config,
        feat_size=128,
        dim_cnn=32
    )
    
    # Needs to skip vector (dim < 3) and nocamera (decode=False)
    assert len(decoder.decoders) == 2
    assert "camera" in decoder.decoders
    assert "lidar" in decoder.decoders
    assert "vector" not in decoder.decoders
    assert "nocamera" not in decoder.decoders

def test_multi_decoder_forward(dummy_obs_space, dummy_obs_config):
    decoder = MultiDecoderNetwork(
        obs_space=dummy_obs_space,
        obs_config=dummy_obs_config,
        feat_size=128,
        dim_cnn=32,
        strides=2,
        padding=1
    )
    
    B, T = 2, 3
    # Inputs is usually TSSM feature (B, T, feat_size)
    inputs = torch.randn(B, T, 128)
    
    out = decoder(inputs)
    
    assert "camera" in out
    assert "lidar" in out
    
    camera_dist = out["camera"]
    # It returns a LaplaceDist, check if we can reconstruct the mode
    recon = camera_dist.mode()
    
    # Should be (B, T, C, H, W)
    assert recon.shape == (B, T, 3, 64, 64)
    
    lidar_dist = out["lidar"]
    recon_lidar = lidar_dist.mode()
    
    # Note: the decoder is initialized with `dim_output_cnn=space.shape[0]` 
    # so for lidar space shape[0] == 1
    assert recon_lidar.shape == (B, T, 1, 64, 64)
