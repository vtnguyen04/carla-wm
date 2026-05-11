import pytest
import torch
from torch_wm.core.base import BaseLoss, BaseCallback

class MockLoss(BaseLoss):
    def name(self): return "mock_loss"
    def compute(self, outputs, batch, **kwargs):
        return torch.tensor(1.0)

def test_base_loss_metrics():
    loss = MockLoss()
    loss.update_metrics({"acc": 0.9})
    metrics = loss.get_metrics()
    assert metrics["mock_loss/acc"] == 0.9
    # Should be cleared
    assert loss.get_metrics() == {}

def test_base_loss_is_enabled():
    loss = MockLoss()
    config = {"modules": {"losses": {"mock_loss": {"enabled": True}}}}
    assert loss.is_enabled(config) is True
    
    config = {"modules": {"losses": {"mock_loss": {"enabled": False}}}}
    assert loss.is_enabled(config) is False

class MockCallback(BaseCallback):
    def name(self): return "mock_cb"
    def is_enabled(self, config): return True
    def on_step_end(self, step, metrics): self.step = step
    def on_epoch_end(self, epoch, metrics): self.epoch = epoch

def test_base_callback():
    cb = MockCallback()
    cb.on_step_end(10, {"loss": 0.5})
    assert cb.step == 10
    cb.on_epoch_end(2, {"loss": 0.5})
    assert cb.epoch == 2
