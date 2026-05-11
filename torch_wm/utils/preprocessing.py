"""Shared observation preprocessing utilities.

Centralizes image normalization (uint8 → [-0.5, 0.5]) and
channel permutation to eliminate duplication across:
- WorldModel.forward()
- WMAgent.preprocess_inputs()
- TwisterAgent._preprocess_obs()
"""

import torch
import numpy as np
from torch_wm.structs import AttrDict


def normalize_image(v: torch.Tensor) -> torch.Tensor:
    """Normalize uint8 images to [-0.5, 0.5] float range.
    
    Args:
        v: Input tensor, possibly uint8.
        
    Returns:
        Float tensor in [-0.5, 0.5] range if input was uint8, unchanged otherwise.
    """
    if v.dtype == torch.uint8:
        return v.float() / 255.0 - 0.5
    return v


def preprocess_obs(obs, device=None, precision="float32"):
    """Preprocess a dict of observations for the world model.
    
    Handles:
    - Tensor conversion (numpy → torch)
    - Device transfer
    - uint8 image normalization to [-0.5, 0.5]
    - Precision casting (float32/float16)
    
    Args:
        obs: Dict of observation tensors, or a single tensor.
        device: Target device (cuda/cpu). If None, tensors stay on current device.
        precision: "float32" or "float16".
        
    Returns:
        Dict of preprocessed tensors, or a single preprocessed tensor wrapped in a dict.
    """
    target_dtype = torch.float16 if precision == "float16" else torch.float32
    
    if isinstance(obs, torch.Tensor):
        if device is not None:
            obs = obs.to(device)
        obs = normalize_image(obs).to(target_dtype)
        return AttrDict({"camera": obs})
    
    if isinstance(obs, dict):
        processed = {}
        for k, v in obs.items():
            if v is None:
                processed[k] = None
                continue
            if not isinstance(v, torch.Tensor):
                v = torch.as_tensor(np.array(v))
            if device is not None:
                v = v.to(device)
            v = normalize_image(v)
            processed[k] = v.to(target_dtype)
        return processed
    
    return obs


def preprocess_obs_for_agent(obs, device):
    """Preprocess observations for TwisterAgent (handles HWC → CHW permutation).
    
    Unlike preprocess_obs(), this also handles the channel permutation 
    needed when data comes from the embodied replay buffer (HWC format).
    
    Args:
        obs: Dict of observation tensors (from replay buffer, HWC format).
        device: Target device.
        
    Returns:
        Dict of preprocessed tensors in CHW format.
    """
    processed = {}
    for k, v in obs.items():
        v = torch.as_tensor(np.array(v)).to(device)
        is_image = (k == 'camera') or (k == 'birdeye_wpt') or (v.dim() >= 3 and v.shape[-1] == 3)
        if is_image:
            if v.dim() == 3: v = v.permute(2, 0, 1)
            elif v.dim() == 4: v = v.permute(0, 3, 1, 2)
            elif v.dim() == 5: v = v.permute(0, 1, 4, 2, 3)
            v = normalize_image(v)
        processed[k] = v
    return processed
