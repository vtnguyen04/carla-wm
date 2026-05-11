import pytest
import pathlib
import torch
import numpy as np
from unittest import mock
from torch_wm.rl.loggers.composer import LogComposer
from torch_wm.structs import AttrDict

def test_log_composer_lifecycle(tmp_path):
    config = AttrDict({
        "batch_size": 1,
        "batch_length": 4,
        "horizon": 10,
        "H": 10,
    })
    
    # Mock sub-loggers to avoid side effects (network, terminal, files)
    with mock.patch("torch_wm.rl.loggers.composer.RichLogger"), \
         mock.patch("torch_wm.rl.loggers.composer.TensorBoardLogger"), \
         mock.patch("torch_wm.rl.loggers.composer.WandBLogger"):
             
        composer = LogComposer(config, tmp_path)
        
        # Test start
        model = mock.MagicMock()
        replay = mock.MagicMock()
        sample = {"obs": np.zeros((64, 64, 3))}
        composer.on_train_start(model, replay, sample)
        
        # Test plan
        composer.on_train_plan(100, 10)
        
        # Test step
        composer.on_step({"loss": 0.5}, 50)
        
        # Test epoch progress
        composer.epoch_progress(1, 10, 100)
        
        # Test epoch end
        composer.on_epoch_end(1, 10, {"loss": 0.5}, 100, 1.0, 100)
        
        # Test deep log with model and batch
        batch = {"camera": torch.zeros(1, 4, 3, 64, 64)}
        composer.on_epoch_end(1, 10, {"loss": 0.5}, 100, 1.0, 100, model=model, batch=batch)
        
        # Test end
        composer.on_train_end(1000, 0.1, "final.pt")
        
        # Test utils
        composer.error("error")
        composer.warning("warning")
        composer.ok("ok")
        composer.on_save("saved")
