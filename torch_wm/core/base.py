"""Core abstractions for WMAgent-CARLA Modular Framework.

Defines base classes and interfaces for all pluggable components
following SOLID principles.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch

class BaseModule(ABC):
    """Abstract base class for all modules in the framework."""

    @abstractmethod
    def name(self) -> str:
        """Return unique name of the module."""
        pass

    @abstractmethod
    def is_enabled(self, config: Dict[str, Any]) -> bool:
        """Check if this module should be active based on config."""
        pass

import torch.nn as nn

class BaseLoss(nn.Module, BaseModule):
    """Abstract base class for all loss functions.

    Follows Strategy Pattern: interchangeable loss algorithms.
    """

    def __init__(self, config: Dict[str, Any] = None, weight: float = 1.0):
        super().__init__()
        self.config = config if config is not None else {}
        self.weight = weight
        self._metrics = {}

    def update_metrics(self, metrics: Dict[str, Any]):
        """Store auxiliary metrics for this loss module."""
        self._metrics.update(metrics)

    def get_metrics(self) -> Dict[str, Any]:
        """Retrieve and reset metrics."""
        m = {f"{self.name()}/{k}": v for k, v in self._metrics.items()}
        self._metrics = {}
        return m

    @abstractmethod
    def compute(
        self,
        model_outputs: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """Compute loss value.
        
        Args:
            model_outputs: Dict of tensors from model forward pass.
            batch: Dict of tensors from dataloader.
            **kwargs: Additional context.
            
        Returns:
            Scalar loss tensor.
        """
        pass

    def is_enabled(self, config: Dict[str, Any]) -> bool:
        """Check if loss is enabled in config."""
        module_config = config.get("modules", {}).get("losses", {}).get(self.name(), {})
        return module_config.get("enabled", False)

class BaseCallback(BaseModule):
    """Abstract base class for training callbacks."""

    @abstractmethod
    def on_step_end(self, step: int, metrics: Dict[str, float]) -> None:
        pass

    @abstractmethod
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        pass
