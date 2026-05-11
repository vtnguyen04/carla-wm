"""Regularizers Package.

Provides modular regularizers for world model training.
All regularizers follow:
- SOLID principles
- Registry Pattern for dynamic registration
- Toggle-able via config
- Independent testability

Usage:
    from torch_wm.modules.regularizers import (
        StraighteningRegularizer,
        VCREGRegularizer,
    )
"""

from .curvature import CurvatureLoss
from .sigreg import SIGRegLoss
from .straightening import StraighteningRegularizer
from .vcreg import VCREGRegularizer

__all__ = [
    'CurvatureLoss',
    'SIGRegLoss',
    'StraighteningRegularizer',
    'VCREGRegularizer',
]
