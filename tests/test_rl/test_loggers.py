import pytest
import pathlib
from torch_wm.rl.loggers.rich_logger import RichLogger
from torch_wm.rl.loggers.composer import LogComposer
from torch_wm.structs import AttrDict

def test_rich_logger_basic():
    logger = RichLogger()
    # Log banner
    logger.print_banner()
    
def test_log_composer():
    config = AttrDict({
        'batch_size': 1,
        'batch_length': 1,
        'horizon': 1,
        'model_lr': 1e-4,
        'actor_lr': 1e-4,
        'critic_lr': 1e-4,
        'grad_clip': 100,
        'stoch_size': 32,
        'discrete': 32,
        'hidden_size': 256,
        'num_blocks_trans': 1,
        'att_context_left': 32,
        'precision': 'float32'
    })
    logdir = pathlib.Path("/tmp/test_log_composer")
    composer = LogComposer(config, logdir)
    
    # Test a few hooks
    composer.on_save("checkpoint.pt")
    composer.warning("test warning")
    composer.ok("test ok")
