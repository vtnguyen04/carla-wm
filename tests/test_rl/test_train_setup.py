import pytest
from unittest.mock import patch, MagicMock
from torch_wm.rl.train import main
import embodied

@patch('embodied.run.train')
@patch('torch_wm.rl.train.WorldModelAgent')
@patch('embodied.Logger')
@patch('carla_env.create_task')
@patch('embodied.replay.Uniform')
def test_train_setup(mock_replay, mock_create_task, mock_logger, mock_agent, mock_run_train):
    mock_agent_inst = MagicMock()
    mock_agent.return_value = mock_agent_inst
    
    mock_env = MagicMock()
    mock_env.obs_space = {'image': MagicMock()}
    mock_env.act_space = {'action': MagicMock()}
    mock_create_task.return_value = (mock_env, {})
    
    with patch('sys.argv', ['train.py', '--logdir', '/tmp/test_train']):
        try:
            main()
        except SystemExit:
            pass
        except Exception:
            pass
    
    assert mock_agent.called
    assert mock_logger.called
