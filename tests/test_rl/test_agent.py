import pytest
import torch
import numpy as np
import embodied
from torch_wm.rl.agent import WorldModelAgent
from torch_wm.structs import AttrDict

@pytest.fixture
def agent_config():
    return embodied.Config({
        'task': 'test_task',
        'device': 'cpu',
        'batch_size': 2,
        'batch_length': 4,
        'stoch_size': 8,
        'discrete': 8,
        'hidden_size': 64,
        'num_blocks_trans': 1,
        'att_context_left': 4,
        'precision': 'float32',
        'model_lr': 1e-4,
        'actor_lr': 1e-4,
        'critic_lr': 1e-4,
        'grad_clip': 100.0,
        'gamma': 0.99,
        'lambda_td': 0.95,
        'dim_cnn': 16,
        'encoder_dim_layers': [16, 32],
        'decoder_dim_layers': [32, 16, 3],
        'modules': {
            'losses': {
                'kl': {'enabled': True, 'weight': 1.0},
                'reconstruction': {'enabled': True, 'weight': 1.0},
                'reward': {'enabled': True, 'weight': 1.0},
                'cpc': {'enabled': False},
                'discount': {'enabled': False}
            }
        }
    })

@pytest.fixture
def spaces():
    obs_space = {
        'camera': embodied.Space(np.uint8, (64, 64, 3)),
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
    }
    act_space = {'action': embodied.Space(np.float32, (15,))}
    return obs_space, act_space

def test_agent_init(agent_config, spaces):
    obs_space, act_space = spaces
    step = embodied.Counter()
    agent = WorldModelAgent(obs_space, act_space, step, agent_config)
    assert agent.model is not None
    assert agent.model.dynamics_model.num_actions == 15

def test_agent_policy(agent_config, spaces):
    obs_space, act_space = spaces
    step = embodied.Counter()
    agent = WorldModelAgent(obs_space, act_space, step, agent_config)
    
    obs = {
        'camera': np.random.randint(0, 256, (64, 64, 3)).astype(np.uint8),
        'is_first': np.array(True)
    }
    
    action, state = agent.policy(obs)
    assert 'action' in action
    assert action['action'].shape == (1, 15)
    assert state is not None

def test_agent_train(agent_config, spaces):
    obs_space, act_space = spaces
    step = embodied.Counter()
    agent = WorldModelAgent(obs_space, act_space, step, agent_config)
    
    B, L = agent_config.batch_size, agent_config.batch_length
    data = {
        'camera': np.random.randint(0, 256, (B, L, 64, 64, 3)).astype(np.uint8),
        'action': np.random.randn(B, L, 15).astype(np.float32),
        'reward': np.random.randn(B, L).astype(np.float32),
        'is_first': np.zeros((B, L), dtype=bool),
        'is_terminal': np.zeros((B, L), dtype=bool),
    }
    data['is_first'][:, 0] = True
    
    outputs, state, metrics = agent.train(data)
    assert 'wm_loss' in metrics
    assert 'actor_loss' in metrics
    assert 'critic_loss' in metrics
    assert metrics['wm_loss'] > 0

def test_agent_save_load(agent_config, spaces):
    obs_space, act_space = spaces
    step = embodied.Counter()
    agent = WorldModelAgent(obs_space, act_space, step, agent_config)
    
    data = agent.save()
    assert 'model' in data
    
    agent.load(data)
    assert agent._updates == 0
