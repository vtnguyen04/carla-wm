import pytest
import torch
from torch_wm.modules.networks.vit import vit_tiny

def test_vit_tiny_forward():
    # Use very small dimensions to save time/memory
    model = vit_tiny(
        patch_size=16,
        img_size=(32, 32),
        num_frames=2,
        tubelet_size=2,
        use_rope=True
    )
    
    # [B, C, T, H, W]
    x = torch.randn(1, 3, 2, 32, 32)
    out = model(x)
    
    # 32/16 = 2. 2*2 = 4 tokens per frame. 2/2 = 1 temporal token.
    # Total tokens = 1 * 4 = 4 tokens? Or is it different?
    # ViT usually flattens tokens.
    assert out.ndim == 3
    assert out.shape[2] == 192
