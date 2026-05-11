import pytest
import torch
from torch_wm.utils.tensors import apply_masks, trunc_normal_

def test_apply_masks():
    x = torch.randn(2, 10, 64)
    # indices to keep: 5 indices per batch
    mask = torch.randint(0, 10, (2, 5))
    out = apply_masks(x, [mask])
    # torch.cat([torch.gather(x, dim=1, index=mask_keep)], dim=0) -> (2, 5, 64)
    assert out.shape == (2, 5, 64)

def test_trunc_normal():
    x = torch.zeros(1000)
    trunc_normal_(x, std=1.0, a=-2.0, b=2.0)
    assert x.min() >= -2.0
    assert x.max() <= 2.0
    assert not torch.allclose(x, torch.zeros(1000))
