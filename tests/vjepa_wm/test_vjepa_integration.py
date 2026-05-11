import pytest
import torch
import sys
import os

# Ensure project root is in path
sys.path.append(os.getcwd())

from torch_wm.core.registry import ModuleRegistry
from torch_wm.models.wm_agent import WMAgent
from torch_wm.structs import AttrDict
from torch_wm.modules.networks import vjepa_encoder
from torch_wm.modules.dynamics import vjepa_dynamics

def test_vjepa_registration():
    """Verify that V-JEPA components are correctly registered in the ModuleRegistry."""
    modules = ModuleRegistry.list_modules()
    assert 'vjepa_encoder' in modules
    assert 'vjepa_predictor' in modules

def test_vjepa_encoder_forward():
    """Verify that the V-JEPA encoder can perform a forward pass with correct output dimensions."""
    encoder_cls = ModuleRegistry.get('vjepa_encoder')
    # Use a small image size and vit_tiny for the test to be efficient
    encoder = encoder_cls(checkpoint_path="", image_size=(64, 64), vit_type="vit_tiny")
    
    B, C, H, W = 2, 3, 64, 64
    x_seq = torch.randn(B, 2, C, H, W)
    
    with torch.no_grad():
        out_dict = encoder(x_seq)
    
    assert "stoch" in out_dict
    assert "logits" in out_dict
    assert "latent" in out_dict
    # (64/16)^2 = 16 patches. vit_tiny embed_dim = 192. 16 * 192 = 3072
    assert out_dict["latent"].shape == (B, 2, 3072)

def test_vjepa_predictor_init():
    """Verify that the V-JEPA predictor initializes correctly."""
    predictor_cls = ModuleRegistry.get('vjepa_predictor')
    # Match vit_tiny embed_dim
    predictor = predictor_cls(num_actions=15, checkpoint_path="", image_size=(64, 64), embed_dim=192)
    
    assert predictor.num_actions == 15
    assert hasattr(predictor, 'predictor')
    assert hasattr(predictor, 'action_encoder')

def test_wm_agent_loading_vjepa():
    """Verify that WMAgent can dynamically load V-JEPA modules via configuration."""
    config = AttrDict({
        "encoder_type": "vjepa_encoder",
        "dynamics_type": "vjepa_predictor",
        "vit_type": "vit_tiny",
        "embed_dim": 192,
        "env_params": {
            "observation": {
                "camera": {"shape": (3, 64, 64), "tssm": True}
            }
        },
        "stoch_size": 32,
        "discrete": 32,
        "hidden_size": 128,
        "num_blocks_trans": 1,
        "att_context_left": 5,
        "image_size": (64, 64),
        "patch_size": 16,
        "tubelet_size": 2,
        "checkpoint_path": "" 
    })
    
    # skip_env=True to avoid CARLA simulator requirement
    agent = WMAgent(env_name="test_vjepa", override_config=config, skip_env=True)
    
    # Check types
    assert agent.encoder_network.__class__.__name__ == "VJEPAEncoderNetwork"
    assert agent.dynamics_model.__class__.__name__ == "VJEPAActionPredictor"
    
    # (64/16)^2 = 16 patches. vit_tiny embed_dim = 192. 16 * 192 = 3072
    assert agent.encoder_network.dim_concat == 16 * 192
