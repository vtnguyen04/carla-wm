import pytest
import torch
import torch.nn as nn
import os
from torch_wm.core.base_model import Model
from torch_wm.core.training_loop import TrainingLoopMixin
from torch_wm.optimizers import Adam
from torch_wm.structs import AttrDict

class MockModel(Model, TrainingLoopMixin):
    def __init__(self, config):
        super().__init__(name="MockModel")
        self.config = config
        self.net = nn.Linear(10, 1)
        self.built = True
        
    def train_step(self, inputs, targets, precision, grad_scaler, accumulated_steps, acc_step, eval_training):
        self.optimizer.zero_grad()
        out = self.net(inputs)
        loss = torch.mean((out - targets)**2)
        loss.backward()
        self.optimizer.step()
        return {"loss": loss}, {"mse": loss.detach()}, 0
    
    def eval_step(self, inputs, targets, verbose):
        with torch.no_grad():
            out = self.net(inputs)
            loss = torch.mean((out - targets)**2)
        return {"loss": loss}, {"mse": loss.detach()}, targets, out

def test_fit_loop(tmp_path):
    config = AttrDict({
        "model_lr": 1e-3,
        "precision": "float32",
        "epochs": 1
    })
    model = MockModel(config)
    model.compile(optimizer=Adam(model.parameters(), lr=1e-3), losses=None)
    model.device = torch.device("cpu")
    
    # Mock dataset as a non-list iterable (like a DataLoader)
    B, D = 2, 10
    class MockLoader:
        def __init__(self, data): self.data = data
        def __iter__(self): return iter(self.data)
        def __len__(self): return len(self.data)
        
    dataset = MockLoader([
        {"inputs": torch.randn(B, D), "targets": torch.randn(B, 1)}
        for _ in range(3)
    ])
    
    # Run fit with all features enabled for coverage
    model.fit(
        dataset_train=dataset,
        dataset_eval=dataset,
        epochs=1,
        steps_per_epoch=2,
        eval_steps=1,
        eval_period_epoch=1,
        saving_period_epoch=1,
        callback_path=str(tmp_path),
        precision=torch.float32,
        verbose_progress_bar=0
    )
    
    assert model.model_step == 2
    assert os.path.exists(tmp_path / "logs")
    # Checkpoints are saved as checkpoints_epoch_...
    assert any("checkpoints_epoch" in f for f in os.listdir(tmp_path))

def test_training_loop_utils():
    config = AttrDict({"model_lr": 1e-3})
    model = MockModel(config)
    from unittest.mock import MagicMock
    writer = MagicMock()
    
    losses = {"loss": torch.tensor(0.5)}
    metrics = {"mse": torch.tensor(0.1)}
    infos = {"lr": 1e-3}
    
    # Test logging and display
    model.log_step(losses, metrics, infos, writer, 1, "tag")
    model.print_step(losses, metrics, "tag")
    
    # Test display_step with mock iterator
    iterator = MagicMock()
    model.display_step(losses, metrics, infos, iterator, 1)
    assert iterator.set_description.called

def test_on_methods():
    model = MockModel({})
    model.on_train_begin()
    model.on_epoch_begin(1)
    # on_epoch_end calls on_step_end
    model.on_epoch_end(False, False, False, None, 1, None, None, None, None, 0, None, False, None)
