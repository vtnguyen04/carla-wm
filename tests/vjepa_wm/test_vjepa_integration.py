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
    # Use a small image size for the test to be efficient
    # checkpoint_path="" to skip loading the massive weights file
    encoder = encoder_cls(checkpoint_path="", image_size=(160, 160))
    
    B, C, H, W = 2, 3, 160, 160
    x = torch.randn(B, C, H, W)
    
    with torch.no_grad():
        # Input can be (B, L, C, H, W)
        x_seq = torch.randn(B, 4, C, H, W)
        out_dict = encoder(x_seq)
    
    assert "stoch" in out_dict
    assert "logits" in out_dict
    assert "latent" in out_dict
    assert out_dict["latent"].shape == (B, 4, 100 * 1024)

def test_vjepa_predictor_init():
    """Verify that the V-JEPA predictor initializes correctly."""
    predictor_cls = ModuleRegistry.get('vjepa_predictor')
    predictor = predictor_cls(num_actions=15, checkpoint_path="", image_size=(160, 160))
    
    assert predictor.num_actions == 15
    assert hasattr(predictor, 'predictor')
    assert hasattr(predictor, 'action_encoder')

def test_wm_agent_loading_vjepa():
    """Verify that WMAgent can dynamically load V-JEPA modules via configuration."""
    config = AttrDict({
        "encoder_type": "vjepa_encoder",
        "dynamics_type": "vjepa_predictor",
        "env_params": {
            "observation": {
                "camera": {"shape": (3, 384, 384), "tssm": True}
            }
        },
        "stoch_size": 32,
        "discrete": 32,
        "hidden_size": 512,
        "num_blocks_trans": 2,
        "att_context_left": 10,
        "image_size": (384, 384),
        "patch_size": 16,
        "tubelet_size": 2,
        "checkpoint_path": "" 
    })
    
    # skip_env=True to avoid CARLA simulator requirement
    agent = WMAgent(env_name="test_vjepa", override_config=config, skip_env=True)
    
    # Check types
    assert agent.encoder_network.__class__.__name__ == "VJEPAEncoderNetwork"
    assert agent.dynamics_model.__class__.__name__ == "VJEPAActionPredictor"
    
    # Check if feature size was correctly inferred (384/16 = 24 -> 24*24=576 tokens -> 576*1024 = 589824)
    # The world model feat_size uses stoch_size * discrete + hidden_size
    # but the encoder output must match what dynamics expect.
    # In V-JEPA dynamics, we simply flattened it.
    assert agent.encoder_network.dim_concat == 24 * 24 * 1024
