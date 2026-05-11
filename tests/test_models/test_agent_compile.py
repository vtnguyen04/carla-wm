import pytest
import torch
from torch_wm.models.wm_agent import WMAgent
from torch_wm.utils.yaml_config import load_yaml_config
import os

@pytest.fixture
def agent_config(dummy_config):
    # Additional required config for agent compile
    dummy_config.stoch_size = 32
    dummy_config.discrete = 32
    dummy_config.hidden_size = 256
    
    # Needs networks config
    dummy_config.encoder = {
        "class": "EncoderNetwork",
        "params": {
            "encoders": {
                "camera": {
                    "image_size": [64, 64],
                    "dim_input_cnn": 3,
                    "dim_output_cnn": 16,
                    "depth_cnn": 2,
                    "mlp_layers": 1,
                    "dim_concat": 128
                }
            }
        }
    }
    dummy_config.decoder = {"class": "DecoderNetwork", "params": {"decoders": {
        "camera": {
            "image_size": [64, 64],
            "dim_input_cnn": 3,
            "dim_output_cnn": 16,
            "depth_cnn": 2
        }
    }}}
    dummy_config.reward = {"class": "RewardNetwork", "params": {}}
    dummy_config.continue_ = {"class": "ContinueNetwork", "params": {}}
    dummy_config.policy = {"class": "PolicyNetwork", "params": {}}
    dummy_config.value = {"class": "ValueNetwork", "params": {}}
    dummy_config.cpc = {"class": "ContrastiveNetwork", "params": {}}
    
    # We will use real yaml configs from rl/config to be safest
    path = os.path.join(os.path.dirname(__file__), "../../torch_wm/rl/config/dreamerv3.yaml")
    import ruamel.yaml as yaml
    model_configs = yaml.YAML(typ='safe').load(open(path))
    from torch_wm.structs import AttrDict
    config = AttrDict(model_configs['defaults'])
    return config

def test_agent_compile(agent_config):
    model = WMAgent(env_name="test_env", override_config=agent_config, skip_env=True)
    
    # Verify core components were instantiated
    assert hasattr(model, "encoder_network")
    assert hasattr(model, "decoder_network")
    assert hasattr(model, "dynamics_model")
    assert hasattr(model, "policy_network")
    
    # Verify inner models are extracted correctly
    assert hasattr(model, "world_model")
    assert hasattr(model, "actor_model")
    assert hasattr(model, "critic_model")

    # Verify parameter lists
    model.compile()
    assert model.optimizer is not None
