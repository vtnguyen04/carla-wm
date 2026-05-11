import pytest
import subprocess
import os
import sys

def test_collect_cli_help():
    """Verify collect.py help command works (basic E2E)."""
    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd()
    result = subprocess.run(
        [sys.executable, 'torch_wm/rl/collect.py', '--help'],
        capture_output=True,
        text=True,
        env=env
    )
    assert result.returncode == 0
    assert 'usage: collect.py' in result.stdout.lower()

def test_train_offline_cli_help():
    """Verify train_offline.py help command works (basic E2E)."""
    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd()
    result = subprocess.run(
        [sys.executable, 'torch_wm/rl/train_offline.py', '--help'],
        capture_output=True,
        text=True,
        env=env
    )
    assert result.returncode == 0
    assert 'usage: train_offline.py' in result.stdout.lower()
