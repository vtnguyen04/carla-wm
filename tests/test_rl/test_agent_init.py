import pytest
import torch
from unittest.mock import MagicMock, patch
from torch_wm.rl.agent import WorldModelAgent
from torch_wm.structs import AttrDict

def test_agent_init_with_expl():
    obs_space = {
        'camera': MagicMock(shape=(64, 64, 3)),
        'vector': MagicMock(shape=(10,))
    }
    act_space = {
        'action': MagicMock(shape=(10,), discrete=False)
    }
    step = MagicMock()
    config = AttrDict({
        'device': 'cpu',
        'task': 'carla_navigation',
        'expl_reward': True,
        'batch_length': 32,
        'env_params': {'action': {'discrete_acc': [-3.0, 0.0, 3.0]}}
    })
    
    agent = WorldModelAgent(obs_space, act_space, step, config)
    assert agent.expl is not None
    assert agent.strategy == 'actor_critic'

def test_agent_policy():
    obs_space = {'camera': MagicMock(shape=(64, 64, 3))}
    act_space = {'action': MagicMock(shape=(10,), discrete=False)}
    step = MagicMock()
    config = AttrDict({'task': 'carla_navigation', 'device': 'cpu'})
    
    agent = WorldModelAgent(obs_space, act_space, step, config)
    
    obs = {
        'camera': torch.randn(1, 3, 64, 64),
        'is_first': torch.tensor([True]),
    }
    state = None
    
    # Use patch context manager correctly
    with patch.object(agent.model, 'policy', return_value=({'action': torch.zeros(1, 10)}, MagicMock())):
        action, state = agent.policy(obs, state, mode='train')
        assert 'action' in action
