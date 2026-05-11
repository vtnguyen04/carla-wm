import pytest
import torch
from torch_wm.core.manager import LossManager, get_config_value
from torch_wm.core.registry import ModuleRegistry
from torch_wm.core.base import BaseLoss
from torch_wm.structs import AttrDict

@ModuleRegistry.register("mock_loss")
class MockLoss(BaseLoss):
    def __init__(self, config=None, weight=1.0):
        super().__init__(config=config, weight=weight)

    def name(self):
        return "mock_loss"

    def compute(self, model_outputs=None, batch=None, **kwargs):
        return torch.tensor(1.0)

def test_get_config_value_attrdict():
    config = AttrDict({"a": {"b": 42}})
    assert get_config_value(config, "a", "b") == 42
    assert get_config_value(config, "a", "c", default=10) == 10

def test_get_config_value_dict():
    config = {"a": {"b": 42}}
    assert get_config_value(config, "a", "b") == 42

def test_loss_manager_init():
    config = AttrDict({
        "modules": {
            "losses": {
                "mock_loss": {"enabled": True, "weight": 2.0}
            }
        }
    })
    manager = LossManager(config)
    assert len(manager.active_losses) == 1
    assert isinstance(manager.active_losses[0], MockLoss)
    assert manager.active_losses[0].weight == 2.0

def test_loss_manager_compute_total_loss():
    config = AttrDict({
        "modules": {
            "losses": {
                "mock_loss": {"enabled": True, "weight": 3.0}
            }
        }
    })
    manager = LossManager(config)
    
    losses = manager.compute_total_loss({}, {})
    assert "mock_loss" in losses
    assert losses["mock_loss"].item() == 1.0
    assert losses["total"].item() == 3.0

def test_loss_manager_metrics():
    config = AttrDict({
        "modules": {
            "losses": {
                "mock_loss": {"enabled": True}
            }
        }
    })
    manager = LossManager(config)
    manager.compute_total_loss({}, {})
    
    metrics = manager.get_metrics()
    assert isinstance(metrics, dict)
