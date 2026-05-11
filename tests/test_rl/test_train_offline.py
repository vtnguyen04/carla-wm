import pytest
import os
import torch
import numpy as np
from unittest import mock
import sys
import pathlib

from torch_wm.rl.train_offline import train_offline, main as train_offline_main
from torch_wm.structs import AttrDict

def test_train_offline_logic(tmp_path):
    # Mock dependencies
    config = AttrDict({
        "steps": 2,
        "batch_size": 1,
        "batch_length": 4,
        "logdir": str(tmp_path),
        "horizon": 10,
    })
    
    agent = mock.MagicMock()
    # Mock dataset iterator to be infinite
    def infinite_dataset():
        while True:
            yield {"obs": {"camera": np.zeros((1, 4, 64, 64, 3))}, "reward": np.zeros((1, 4))}
            
    agent.dataset.return_value = infinite_dataset()
    agent.train.return_value = ({}, {}, {"wm_loss": torch.tensor(0.5)})
    agent._preprocess_obs.return_value = {"camera": torch.zeros(1, 4, 3, 64, 64)}
    agent.model = mock.MagicMock()
    agent.model.state_dict.return_value = {}
    
    replay = mock.MagicMock()
    # Ensure steps_per_epoch = 1
    replay.__len__.return_value = 4 # batch_size * batch_length
    replay.dataset = iter([])
    
    logger = mock.MagicMock()
    logger.logdir = tmp_path
    # Mock context manager for progress bar
    mock_progress = mock.MagicMock()
    logger.epoch_progress.return_value.__enter__.return_value = mock_progress
    
    # Run training logic
    train_offline(agent, replay, config, logger)
    
    assert logger.on_train_plan.called
    assert logger.on_epoch_end.called
    assert logger.on_train_end.called

def test_load_replay(tmp_path):
    from torch_wm.rl.train_offline import load_replay
    config = AttrDict({"batch_length": 4, "replay_size": 100})
    replay = load_replay(tmp_path, config)
    assert hasattr(replay, 'dataset')
def test_train_offline():
    from torch_wm.rl.train_offline import train_offline
    import embodied
    from unittest import mock

    # Use a real simple model for state_dict to avoid pickle issues
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)
        def state_dict(self, *args, **kwargs):
            return {"linear.weight": torch.tensor([[1.0]])}

    agent = mock.MagicMock()
    agent.model = SimpleModel()
    agent.step = embodied.Counter()
    # train() must return (outs, state, metrics)
    agent.train.return_value = ({"loss": 1.0}, {}, {"metrics": 1.0})
    replay = mock.MagicMock()
    replay.dataset.return_value = iter([{"camera": np.zeros((4, 64, 64, 3))}])
    logger = mock.MagicMock()
    config = AttrDict({
        "train_steps": 1, 
        "eval_every": 10, 
        "log_every": 1, 
        "save_every": 10, 
        "batch_length": 4,
        "batch_size": 1,
        "steps": 1
    })

    with mock.patch("embodied.Timer"):
        with mock.patch("embodied.Metrics"):
            train_offline(agent, replay, config, logger)
    assert agent.train.called


def test_train_offline_main_setup(tmp_path):
    test_args = [
        "--logdir", str(tmp_path),
        "--replay_dir", str(tmp_path / "replay"),
        "--steps", "10",
        "--task", "dummy_task"
    ]
    
    # Create a real dummy config file to avoid open() mocking issues
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "dreamerv3.yaml"
    
    yaml_content = """
defaults:
    logdir: "/tmp"
    batch_size: 1
    batch_length: 4
    steps: 10
    task: "dummy_task"
    horizon: 10
    model_lr: 1e-4
    actor_lr: 8e-5
    critic_lr: 8e-5
    grad_clip: 100.0
    gamma: 0.99
    lambda_td: 0.95
    stoch_size: 32
    discrete: 32
    hidden_size: 256
    num_blocks_trans: 1
    att_context_left: 32
    precision: "float32"
    num_actions: 15
    run:
        steps: 10
        from_checkpoint: ''
        log_zeros: False
        log_keys_video: ['camera']
"""
    config_file.write_text(yaml_content)
    
    # Mock sys.argv
    with mock.patch("sys.argv", ["train_offline.py"] + test_args):
        with mock.patch("pathlib.Path.resolve") as mock_resolve:
            mock_path = mock.MagicMock()
            mock_path.parent = tmp_path
            mock_resolve.return_value = mock_path
            
            with mock.patch("torch_wm.rl.train_offline.load_replay") as mock_load:
                mock_replay = mock.MagicMock()
                mock_replay.__len__.return_value = 10
                mock_replay.dataset.return_value = iter([{"camera": np.zeros((4, 64, 64, 3))}])
                mock_load.return_value = mock_replay
                
                with mock.patch("torch_wm.rl.train_offline.train_offline") as mock_train:
                    with mock.patch("torch_wm.rl.agent.WorldModelAgent"):
                        train_offline_main()
                        assert mock_train.called
