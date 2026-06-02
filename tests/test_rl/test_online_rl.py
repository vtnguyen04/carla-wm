import pytest
import os
import torch
import numpy as np
from unittest import mock
import sys
import ruamel.yaml as yaml

from torch_wm.rl._setup_path import setup; setup()
import embodied

from torch_wm.rl.train import main as train_main
from torch_wm.rl.eval import main as eval_main

# Dummy environment that behaves like CARLA for Embodied
class DummyEnv:
    def __init__(self, *args, **kwargs):
        self._step = 0
        
        # Define mock spaces mimicking what gym would provide
        class MockSpace:
            def __init__(self, shape, dtype):
                self.shape = shape
                self.dtype = dtype
                self.low = np.zeros(shape, dtype=dtype)
                self.high = np.ones(shape, dtype=dtype) * 255 if dtype == np.uint8 else np.ones(shape, dtype=dtype)
            def sample(self):
                if self.dtype == np.uint8:
                    return np.zeros(self.shape, dtype=self.dtype)
                return np.zeros(self.shape, dtype=self.dtype)

        self.observation_space = type('obj', (object,), {
            'spaces': {
                "camera": MockSpace((64, 64, 3), np.uint8),
                "vector": MockSpace((10,), np.float32),
            }
        })()
        
        self.action_space = type('obj', (object,), {
            'spaces': {
                "action": MockSpace((15,), np.float32)
            }
        })()

    def step(self, action):
        self._step += 1
        obs = {
            "camera": np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
            "vector": np.random.randn(10).astype(np.float32),
        }
        reward = np.float32(1.0)
        done = self._step >= 10
        info = {}
        if done:
            self._step = 0
        return obs, reward, done, info

    def reset(self, **kwargs):
        self._step = 0
        obs = {
            "camera": np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
            "vector": np.random.randn(10).astype(np.float32),
        }
        return obs

    def info(self):
        return {}

    def close(self):
        pass

def mock_create_task(name, other):
    # Returns a dummy env and empty config dict
    return DummyEnv(), {}

def test_online_train(tmp_path):
    test_args = [
        "--logdir", str(tmp_path / "train"),
        "--task", "dummy_env",
        "--batch_size", "2",
        "--batch_length", "4",
        "--steps", "8",
        "--eval_every", "1000",
        "--log_every", "1000",
        "--save_every", "1000",
    ]
    
    yaml_content = """
defaults:
    logdir: "/tmp"
    batch_size: 1
    batch_length: 4
    steps: 10
    task: "dummy_env"
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
    eval_every: 1000
    log_every: 1000
    save_every: 1000
    train_ratio: 16
    train_fill: 500
    run:
        steps: 10
        from_checkpoint: 'dummy_ckpt'
        log_zeros: False
        log_keys_video: ['camera']
"""
    import pathlib
    with mock.patch("pathlib.Path.resolve") as mock_resolve:
        mock_resolve.return_value.parent = pathlib.Path(".")
        with mock.patch("builtins.open", mock.mock_open(read_data="defaults: {}")):
            import ruamel.yaml as yaml_lib
            mock_config = yaml_lib.YAML(typ='safe').load(yaml_content)
            with mock.patch("ruamel.yaml.YAML.load", return_value=mock_config):
                with mock.patch("sys.argv", ["train.py"] + test_args):
                    os.environ["WANDB_MODE"] = "offline"
                    with mock.patch("carla_env.create_task", side_effect=mock_create_task):
                        train_main()

def test_online_eval(tmp_path):
    test_args = [
        "--wm_agent.logdir", str(tmp_path / "eval"),
        "--task", "dummy_env",
        "--wm_agent.run.steps", "8"
    ]
    
    yaml_content = """
defaults:
    logdir: "/tmp"
    batch_size: 1
    batch_length: 4
    steps: 10
    task: "dummy_env"
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
    wandb: False
    run:
        steps: 10
        from_checkpoint: 'dummy_ckpt'
        log_zeros: False
        log_keys_video: ['camera']
    wrapper:
        length: 4000
        reset: True
        discretize: 0
        checks: False
        repeat: 1
wm_agent:
    logdir: "/tmp"
"""

    
    import pathlib
    with mock.patch("pathlib.Path.resolve") as mock_resolve:
        mock_resolve.return_value.parent = pathlib.Path(".")
        with mock.patch("builtins.open", mock.mock_open(read_data="defaults: {}")):
            import ruamel.yaml as yaml_lib
            mock_config = yaml_lib.YAML(typ='safe').load(yaml_content)
            with mock.patch("ruamel.yaml.YAML.load", return_value=mock_config):
                with mock.patch.dict("sys.modules", {"tensorflow": mock.MagicMock(), "tensorflow.compat.v1": mock.MagicMock()}):
                    with mock.patch("sys.argv", ["eval.py"] + test_args):
                        os.environ["WANDB_MODE"] = "disabled"
                        with mock.patch("wandb.init"):
                            with mock.patch("carla_env.create_task", side_effect=mock_create_task):
                                with mock.patch("embodied.Checkpoint.load", return_value=None):
                                    eval_main()
def test_online_collect(tmp_path):
    from torch_wm.rl.collect import main as collect_main
    test_args = [
        "collect.py",
        "--logdir", str(tmp_path / "collect"),
        "--task", "dummy_env",
        "--steps", "8",
        "--device", "cpu",
    ]
    
    with mock.patch.object(sys, "argv", test_args):
        os.environ["WANDB_MODE"] = "offline"
        with mock.patch("carla_env.create_task", side_effect=mock_create_task):
            collect_main()
