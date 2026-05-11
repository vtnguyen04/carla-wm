import pytest
import torch
from torch_wm.modules.layers.conv_transpose_2d import ConvTranspose2d
from torch_wm.modules.layers.dropout import Dropout
from torch_wm.modules.layers.ffn import MLP, SwiGLUFFN
from torch_wm.modules.layers.linear import Linear

def test_conv_transpose_2d():
    layer = ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1)
    x = torch.randn(1, 16, 4, 4)
    out = layer(x)
    assert out.shape == (1, 8, 8, 8)

def test_dropout():
    layer = Dropout(0.5)
    x = torch.randn(1, 10)
    out = layer(x)
    assert out.shape == (1, 10)

def test_mlp():
    layer = MLP(128, 512, drop=0.1)
    x = torch.randn(1, 10, 128)
    out = layer(x)
    assert out.shape == (1, 10, 128)

def test_swiglu_ffn():
    layer = SwiGLUFFN(128, 512)
    x = torch.randn(1, 10, 128)
    out = layer(x)
    assert out.shape == (1, 10, 128)

def test_linear():
    layer = Linear(64, 32)
    x = torch.randn(1, 64)
    out = layer(x)
    assert out.shape == (1, 32)
