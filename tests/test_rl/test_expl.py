import pytest
import torch
from torch_wm.rl.expl import Disag
from torch_wm.structs import AttrDict

def test_disag_init():
    config = AttrDict({
        'disag_models': 3,
        'hidden_size': 64,
        'deter_size': 128,
        'num_actions': 10,
        'expl_lr': 1e-4
    })
    expl = Disag(config)
    
    # Test forward
    traj_feat = torch.randn(2, 5, 64)
    actions = torch.randn(2, 5, 10)
    reward = expl(traj_feat, actions)
    assert reward.shape == (2, 5)
    
    # Test train_step
    targets = torch.randn(2, 5, 128)
    metrics = expl.train_step(traj_feat, actions, targets)
    assert 'loss_disag' in metrics
    assert metrics['loss_disag'] > 0
