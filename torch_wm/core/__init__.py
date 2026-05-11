"""Core package for WMAgent-CARLA Modular Framework."""

from .base import BaseModule, BaseLoss, BaseCallback
from .registry import ModuleRegistry
from .manager import LossManager

__all__ = [
    "BaseModule",
    "BaseLoss",
    "BaseCallback",
    "ModuleRegistry",
    "LossManager",
]
