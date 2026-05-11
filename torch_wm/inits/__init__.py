import torch
import torch.nn as nn
import numpy as np

def dreamerv3_normal(tensor, std=None):
    """Special normal initialization scaled by fan-in."""
    if std is None:
        try:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(tensor)
            std = 1.0 / np.sqrt(max(1, fan_in))
        except:
            std = 0.02
    with torch.no_grad():
        tensor.normal_(0, std)
    return tensor

def xavier_uniform(tensor, gain=1.0):
    return nn.init.xavier_uniform_(tensor, gain=gain)

def xavier_normal(tensor, gain=1.0):
    return nn.init.xavier_normal_(tensor, gain=gain)

def zeros(tensor):
    return nn.init.constant_(tensor, 0.0)

def ones(tensor):
    return nn.init.constant_(tensor, 1.0)

def scaled_uniform(tensor, scale=1.0):
    """Uniform initialization scaled by fan-in."""
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(tensor)
    std = scale / np.sqrt(fan_in)
    return nn.init.uniform_(tensor, -std, std)

def normal_02(tensor):
    """Normal initialization with std=0.02."""
    return nn.init.normal_(tensor, std=0.02)

# Dictionary for registry-like lookup used in linear.py and conv2d.py
init_dict = {
    "default": xavier_uniform,
    "zeros": zeros,
    "ones": ones,
    "xavier_uniform": xavier_uniform,
    "xavier_normal": xavier_normal,
    "dreamerv3_normal": dreamerv3_normal,
    "scaled_uniform": scaled_uniform,
    "normal_02": normal_02,
}
