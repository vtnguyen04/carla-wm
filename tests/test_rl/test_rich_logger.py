import pytest
import torch
from unittest.mock import MagicMock
from torch_wm.rl.loggers.rich_logger import RichLogger

def test_rich_logger_full_coverage():
    logger = RichLogger()
    
    # Init shouldn't crash
    assert logger is not None
    
    # Test properties
    logger.print_banner()
    logger.print_hardware()
    
    mock_config = MagicMock(batch_size=2, batch_length=10, horizon=15, model_lr=1e-4, actor_lr=1e-4, critic_lr=1e-4, grad_clip=10.0, stoch_size=32, discrete=True, hidden_size=64, num_blocks_trans=2, att_context_left=10, precision="float32")
    logger.print_config(mock_config)
    
    mock_model = MagicMock()
    mock_model.encoder_network.encoders = {"camera": MagicMock(dim_input_cnn=3, image_size=(64, 64), dim_concat=32)}
    mock_model.encoder_network._tssm_branches = ["camera"]
    mock_model.encoder_network.dim_concat = 100
    mock_model.encoder_network.dim_signal = 0
    mock_model.parameters = lambda: [torch.ones(10)]
    mock_model.config = mock_config
    
    logger.print_architecture(mock_model)
    logger.print_losses({"modules": {"losses": {"kl": {"enabled": True, "weight": 1.0}}}})
    logger.print_replay([1], {"camera": torch.ones(2, 3)})
    logger.print_plan(100, 10, 2, 10, 15)
    
    with logger.epoch_progress(1, 10, 100) as p:
        pass
    
    logger.print_epoch(1, 10, {"wm_loss": 0.5}, "1m", 100)
    logger.print_gpu_status(10.0, 60.0)
    logger.print_save("check")
    logger.print_summary(100, 0.5, "path", "path")
    logger.print_error("err")
    logger.print_warning("warn")
    logger.print_ok("ok")
