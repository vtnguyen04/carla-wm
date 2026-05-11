import pytest
import torch
import numpy as np
from torch_wm.utils.preprocessing import normalize_image, preprocess_obs, preprocess_obs_for_agent
from torch_wm.structs import AttrDict

def test_normalize_image():
    # Test uint8 normalization
    v_uint8 = torch.tensor([0, 127, 255], dtype=torch.uint8)
    v_norm = normalize_image(v_uint8)
    assert v_norm.dtype == torch.float32
    assert torch.allclose(v_norm, torch.tensor([-0.5, -0.00196078, 0.5]))

    # Test float stays unchanged
    v_float = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float32)
    v_out = normalize_image(v_float)
    assert v_out.dtype == torch.float32
    assert torch.allclose(v_out, v_float)

def test_preprocess_obs_single_tensor():
    # Test processing a single tensor
    obs = torch.tensor([0, 127, 255], dtype=torch.uint8)
    processed = preprocess_obs(obs, device="cpu", precision="float32")
    
    assert isinstance(processed, AttrDict)
    assert "camera" in processed
    assert processed.camera.dtype == torch.float32
    assert torch.allclose(processed.camera, torch.tensor([-0.5, -0.00196078, 0.5]))
    
    # Test float16 precision
    processed_fp16 = preprocess_obs(obs, device="cpu", precision="float16")
    assert processed_fp16.camera.dtype == torch.float16

def test_preprocess_obs_dict():
    # Test processing a dict of numpy arrays and tensors
    obs = {
        "camera": np.array([0, 255], dtype=np.uint8),
        "vector": torch.tensor([1.0, 2.0], dtype=torch.float32)
    }
    
    processed = preprocess_obs(obs, device="cpu", precision="float32")
    assert isinstance(processed, dict)
    assert "camera" in processed
    assert "vector" in processed
    assert processed["camera"].dtype == torch.float32
    assert torch.allclose(processed["camera"], torch.tensor([-0.5, 0.5]))
    assert processed["vector"].dtype == torch.float32
    assert torch.allclose(processed["vector"], torch.tensor([1.0, 2.0]))

def test_preprocess_obs_passthrough():
    # Test when obs is neither dict nor tensor
    obs = [1, 2, 3]
    processed = preprocess_obs(obs)
    assert processed == [1, 2, 3]

def test_preprocess_obs_for_agent():
    # Test HWC to CHW permutation and normalization
    # Shape: (B, H, W, C) -> (1, 64, 64, 3)
    camera_b_hwc = np.ones((1, 64, 64, 3), dtype=np.uint8) * 255
    # Shape: (B, T, H, W, C) -> (1, 2, 64, 64, 3)
    birdeye_bt_hwc = np.zeros((1, 2, 64, 64, 3), dtype=np.uint8)
    # Shape: (H, W, C) -> (64, 64, 3)
    single_img_hwc = np.ones((64, 64, 3), dtype=np.uint8) * 127
    
    obs = {
        "camera": camera_b_hwc,
        "birdeye_wpt": birdeye_bt_hwc,
        "single_img": single_img_hwc,
        "vector": np.array([1.0, 2.0])
    }
    
    processed = preprocess_obs_for_agent(obs, device="cpu")
    
    # Check permutations
    assert processed["camera"].shape == (1, 3, 64, 64) # B, C, H, W
    assert processed["birdeye_wpt"].shape == (1, 2, 3, 64, 64) # B, T, C, H, W
    assert processed["single_img"].shape == (3, 64, 64) # C, H, W
    assert processed["vector"].shape == (2,)
    
    # Check normalization and types
    assert processed["camera"].dtype == torch.float32
    assert torch.allclose(processed["camera"], torch.tensor(0.5))
    
    assert processed["birdeye_wpt"].dtype == torch.float32
    assert torch.allclose(processed["birdeye_wpt"], torch.tensor(-0.5))
    
    assert processed["vector"].dtype == torch.float64 # Preserves float64 from numpy
    assert torch.allclose(processed["vector"], torch.tensor([1.0, 2.0], dtype=torch.float64))
