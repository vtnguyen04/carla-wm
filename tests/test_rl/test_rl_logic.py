import pytest
import torch
import numpy as np
from torch_wm.rl import collect, eval as rl_eval, train_offline
from torch_wm.structs import AttrDict

def test_train_offline_logic():
    # Test if we can find the core training function
    assert hasattr(train_offline, 'train_offline')

def test_agent_integration():
    from torch_wm.models.unified_agent import UnifiedAgent
    config = AttrDict({
        "stoch_size": 32, "discrete": 32, "hidden_size": 256,
        "num_blocks_trans": 1, "att_context_left": 32,
        "precision": "float32", "model_lr": 1e-4, "actor_lr": 8e-5, "critic_lr": 8e-5,
        "num_actions": 15,
        "env_params": {"observation": {"enabled": ["camera"], "camera": {"shape": [3, 64, 64]}}}
    })
    # Default is actor_critic
    agent = UnifiedAgent(env_name="test", override_config=config)
    assert agent.strategy_name == "actor_critic"

    # Switch to mpc with MINIMAL iterations for speed
    mpc_config = config.copy()
    mpc_config["agent_strategy"] = "mpc"
    mpc_config["samples"] = 2
    mpc_config["topk"] = 1
    mpc_config["iterations"] = 1
    mpc_config["horizon"] = 1
    
    agent_mpc = UnifiedAgent(env_name="test", override_config=mpc_config)
    assert agent_mpc.strategy_name == "mpc"
    assert hasattr(agent_mpc, "planner")

    # Test act call
    obs = {"camera": torch.randn(1, 3, 64, 64)}
    is_first = torch.tensor([True])

    action, state = agent_mpc.act(obs, is_first)
    assert isinstance(action, torch.Tensor)
