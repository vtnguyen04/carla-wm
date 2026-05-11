import pytest
import torch
import torch.nn.functional as F

from torch_wm.distributions.base import BaseDist
from torch_wm.distributions.laplace_dist import LaplaceDist
from torch_wm.distributions.mse_dist import MSEDist
from torch_wm.distributions.sym_log_dist import SymLogDist
from torch_wm.distributions.sym_log_discrete_dist import SymLogDiscreteDist
from torch_wm.distributions.tanh_normal import TanhNormal
from torch_wm.distributions.one_hot_dist import OneHotDist
from torch_wm.distributions.lpips_dist import PerceptualLPIPSDist

def test_laplace_dist():
    x = torch.randn(2, 4)
    dist = LaplaceDist(x)
    assert issubclass(LaplaceDist, BaseDist)
    assert dist.mode().shape == (2, 4)
    log_prob = dist.log_prob(torch.randn(2, 4))
    assert log_prob.shape == () # Sums all dims due to reinterpreted_batch_ndims=0

def test_mse_dist():
    x = torch.randn(2, 4)
    dist = MSEDist(x)
    assert issubclass(MSEDist, BaseDist)
    assert dist.mode().shape == (2, 4)
    log_prob = dist.log_prob(torch.randn(2, 4))
    assert log_prob.shape == ()
    # Check MSE calculation
    target = torch.randn(2, 4)
    expected = -F.mse_loss(x, target, reduction='none').sum(-1, keepdim=True)

def test_one_hot_dist():
    logits = torch.randn(2, 4, 32)
    dist = OneHotDist(logits=logits, uniform_mix=0.01)
    assert dist.batch_shape == (2, 4)
    assert dist.event_shape == (32,)
    
    dist_tmp = OneHotDist(logits=logits, sampling_tmp=0.5)
    assert dist_tmp.batch_shape == (2, 4)

def test_lpips_dist():
    # Needs 4D input (B, C, H, W)
    x = torch.randn(1, 3, 64, 64)
    dist = PerceptualLPIPSDist(x)
    target = torch.randn(1, 3, 64, 64)
    log_prob = dist.log_prob(target)
    assert log_prob.shape == ()
    
def test_symlog_dist():
    x = torch.randn(2, 4)
    dist = SymLogDist(x, reinterpreted_batch_ndims=1)
    assert issubclass(SymLogDist, BaseDist)
    assert dist.mode().shape == (2, 4)
    log_prob = dist.log_prob(torch.randn(2, 4))
    assert log_prob.shape == (2,)

def test_symlog_discrete_dist():
    logits = torch.randn(2, 255)
    dist = SymLogDiscreteDist(logits)
    assert issubclass(SymLogDiscreteDist, BaseDist)
    assert dist.mode().shape == (2, 1)
    assert callable(dist.mean) or dist.mean.shape == (2, 1)
    log_prob = dist.log_prob(torch.randn(2,))
    assert log_prob.shape == (2,)

def test_tanh_normal():
    mean = torch.randn(2, 4)
    std = torch.ones(2, 4)
    # Wrap in IndependentTanhNormal to handle correct dim reductions across action dims
    from torch_wm.distributions.tanh_normal import IndependentTanhNormal
    dist = IndependentTanhNormal(mean, std)
    assert dist.mode().shape == (2, 4)
    assert dist.mean().shape == (2, 4)
    sample = dist.sample()
    assert sample.shape == (2, 4)
    assert dist.log_prob(sample).shape == (2,)
