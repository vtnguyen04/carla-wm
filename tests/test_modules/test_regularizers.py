import pytest
import torch

from torch_wm.modules.regularizers.straightening import StraighteningRegularizer
from torch_wm.modules.regularizers.vcreg import VCREGRegularizer

def test_straightening():
    reg = StraighteningRegularizer(config={"weight": 1.0}, weight=1.0)
    # L=5, D=3
    trajectory = torch.randn(2, 5, 3)
    loss = reg.compute({'post': {'deter': trajectory}}, {})
    assert loss.dim() == 0
    
def test_vcreg():
    reg = VCREGRegularizer(config={'std_weight': 1.0, 'cov_weight': 1.0}, weight=1.0)
    feats = torch.randn(10, 64)
    loss = reg.compute({'post': {'stoch': feats}}, {})
    assert loss.dim() == 0

from torch_wm.modules.regularizers.curvature import CurvatureLoss
from torch_wm.modules.regularizers.sigreg import SIGRegLoss

def test_curvature():
    reg = CurvatureLoss(config={'weight': 1.0}, weight=1.0)
    # L=5, D=3
    trajectory = torch.randn(2, 5, 3)
    loss = reg.compute({'latent': trajectory}, {})
    assert loss.dim() == 0

def test_sigreg():
    reg = SIGRegLoss(config={'weight': 1.0}, weight=1.0)
    trajectory = torch.randn(2, 5, 3)
    loss = reg.compute({'latent': trajectory}, {})
    assert loss.dim() == 0
