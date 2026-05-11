import pytest
import torch
import numpy as np
from torch_wm.utils.collate_fn import CollateFn

def test_collate_fn_basic():
    cf = CollateFn()
    batch = [
        {'obs': torch.zeros(3, 32, 32), 'action': torch.tensor([1.0])},
        {'obs': torch.ones(3, 32, 32), 'action': torch.tensor([0.0])}
    ]
    collated, targets = cf(batch)
    assert isinstance(collated['obs'], torch.Tensor)
    assert collated['obs'].shape == (2, 3, 32, 32)
    assert collated['action'].shape == (2, 1)

def test_collate_fn_numpy():
    cf = CollateFn()
    batch = [
        {'obs': np.zeros((3, 32, 32))},
        {'obs': np.ones((3, 32, 32))}
    ]
    collated, targets = cf(batch)
    assert isinstance(collated['obs'], torch.Tensor)
    assert collated['obs'].shape == (2, 3, 32, 32)

def test_collate_fn_nested():
    cf = CollateFn()
    batch = [
        {'obs': {'image': np.zeros((3, 32, 32))}},
        {'obs': {'image': np.ones((3, 32, 32))}}
    ]
    collated, targets = cf(batch)
    assert collated['obs']['image'].shape == (2, 3, 32, 32)

def test_collate_fn_non_list():
    cf = CollateFn()
    x = {"a": 1}
    assert cf(x) == x
