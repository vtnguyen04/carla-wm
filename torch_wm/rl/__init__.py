"""World Model RL Package — Unified under torch_wm.

Contains the training pipeline, agent, loggers, and embodied interface.

Usage:
    from torch_wm.rl.agent import WorldModelAgent
    from torch_wm.rl.loggers import LogComposer
"""

# Lazy imports to avoid requiring 'embodied' at package load time
__all__ = ['WorldModelAgent']


def __getattr__(name):
    """Lazy import WorldModelAgent only when accessed."""
    if name == 'WorldModelAgent':
        from torch_wm.rl.agent import WorldModelAgent
        return WorldModelAgent
    raise AttributeError(f"module 'torch_wm.rl' has no attribute {name!r}")
