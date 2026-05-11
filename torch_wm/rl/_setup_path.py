"""Path setup for RL entry points.

Adds the project root to sys.path so that:
- `import torch_wm` works
- `import embodied` works (Standalone library at project root)
"""

import pathlib
import sys


def setup():
    """Configure sys.path for RL entry points. Idempotent."""
    root = str(pathlib.Path(__file__).resolve().parent.parent.parent)
    
    if root not in sys.path:
        sys.path.insert(0, root)

