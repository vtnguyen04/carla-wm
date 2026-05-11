# Copyright 2025, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch
import torch
import torch.nn as nn

# NeuralNets
from torch_wm import modules
from torch_wm import distributions
from torch_wm.structs import AttrDict

class DecoderNetwork(nn.Module):

    def __init__(
        self,
        feat_size=32*32+512,
        dim_cnn=32,
        dim_output_cnn=3,
        act_fun=nn.SiLU,
        weight_init="dreamerv3_normal",
        bias_init="zeros",
        cnn_norm={"class": "LayerNorm", "params": {"eps": 1e-3}},
        dist_weight_init="zeros",
        dist_bias_init="zeros",
        image_size=(64, 64),
        **kwargs
    ):
        super(DecoderNetwork, self).__init__()

        self.feat_size = feat_size
        self.dim_output_cnn = dim_output_cnn
        self.dim_cnn = dim_cnn
        self.image_size = image_size


        # Architecture configuration — 5-layer decoder for sufficient capacity
        # to invert deep pretrained encoder features (ResNet18 → 512-dim)
        self.dim_layers = kwargs.get("decoder_dim_layers", [8*dim_cnn, 4*dim_cnn, 2*dim_cnn, 1*dim_cnn, dim_output_cnn])
        self.kernel_size = kwargs.get("kernel_size", 4)
        self.strides = kwargs.get("strides", 2)
        self.padding = kwargs.get("padding", 1)

        # Infer dimensions for projection and deconvolution
        self.dim_root, self.dim_proj_out = self._infer_decoder_shapes()

        # CNN proj: Maps latent features to the spatial root of the deconvolution
        self.proj = modules.Linear(feat_size, self.dim_proj_out, weight_init="xavier_uniform", bias_init="zeros")

        # CNN: Deconvolutional layers to upscale to target image_size
        self.cnn = modules.ConvTransposeNeuralNetwork(
            dim_input=self.dim_root[0],
            dim_layers=self.dim_layers,
            kernel_size=self.kernel_size,
            strides=self.strides,
            act_fun=[act_fun for _ in range(len(self.dim_layers)-1)] + [None],
            weight_init=[weight_init for _ in range(len(self.dim_layers)-1)] + [dist_weight_init],
            bias_init=[bias_init for _ in range(len(self.dim_layers)-1)] + [dist_bias_init],
            norm=[cnn_norm for _ in range(len(self.dim_layers)-1)] + [None],
            bias=[cnn_norm is None for _ in range(len(self.dim_layers)-1)] + [True],
            padding=self.padding,
            output_padding=0,
            channels_last=False
        )

    def _infer_decoder_shapes(self):
        """Calculate the spatial root size and projection dimension required to reach image_size."""
        h, w = self.image_size
        n_layers = len(self.dim_layers)

        # Reverse strides to find the root size (e.g., 128 -> 64 -> 32 -> 16 -> 8)
        for _ in range(n_layers):
            h = h // self.strides if isinstance(self.strides, int) else h // self.strides[_]
            w = w // self.strides if isinstance(self.strides, int) else w // self.strides[_]

        dim_root = (self.dim_layers[0] if n_layers > 0 else self.dim_output_cnn, h, w)
        dim_proj_out = dim_root[0] * dim_root[1] * dim_root[2]
        return dim_root, dim_proj_out

    def forward_cnn(self, x):

        # (B, N, D) -> (B, N, C * 4 * 4)
        x = self.proj(x)

        # (B, N, C * H * W) -> (B * N, C, H, W)
        shape = x.shape
        x = x.reshape((-1,) + self.dim_root)

        # (B * N, C, H, W) -> (B * N, 3, Target_H, Target_W)
        x = self.cnn(x)

        # (B * N, 3, Target_H, Target_W) -> (B, N, 3, Target_H, Target_W)
        x = x.reshape(shape[:-1] + x.shape[1:])

        # L1 (Laplace) Loss: Bền vững tuyệt đối. Dùng agg="sum" để gradient từng pixel là 1.0, không bị triệt tiêu khi truyền ngược.
        obs_dist = distributions.LaplaceDist(x, agg="sum", reinterpreted_batch_ndims=3)
        return obs_dist

    def forward(self, inputs):
        # Outputs
        outputs = self.forward_cnn(inputs)
        return outputs

class MultiDecoderNetwork(nn.Module):
    """
    Decodes stochastic representations into multiple observation maps based on configuration.
    """
    def __init__(
        self,
        obs_config,
        obs_space,
        feat_size=32*32+512,
        dim_cnn=32,
        dim_output_cnn=3,
        act_fun=nn.SiLU,
        weight_init="dreamerv3_normal",
        bias_init="zeros",
        cnn_norm={"class": "LayerNorm", "params": {"eps": 1e-3}},
        dist_weight_init="xavier_uniform",
        dist_bias_init="zeros",
        **kwargs
    ):
        super(MultiDecoderNetwork, self).__init__()

        self.decoders = nn.ModuleDict()

        # Dynamically create decoders for each enabled sensor with decode=True
        if obs_space is None:
            # Fallback if obs_space is entirely missing
            obs_space = {}
            # If no 'enabled' list, use all keys that look like observations
            enabled_keys = obs_config.get("enabled")
            if enabled_keys is None:
                enabled_keys = [k for k, v in obs_config.items() if isinstance(v, (dict, AttrDict))]
                
            for k in enabled_keys:
                cfg = obs_config.get(k, {})
                shape = cfg.get("shape", (3, 64, 64))
                class SpaceStub:
                    def __init__(self, s): self.shape = s
                obs_space[k] = SpaceStub(shape)

        for name, space in obs_space.items():
            if len(space.shape) < 3:
                continue

            config = obs_config.get(name, AttrDict())

            # Skip sensors marked as encode-only (decode=False)
            if not config.get("decode", True):
                continue

            # Ensure config is AttrDict
            if not isinstance(config, AttrDict):
                config = AttrDict(config)

            # Extract image size
            image_size = (space.shape[1], space.shape[2])

            # Create sub-decoder for this sensor branch
            # Filter image_size from kwargs to avoid duplicate argument error
            filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'image_size'}
            
            self.decoders[name] = DecoderNetwork(
                feat_size=feat_size,
                dim_cnn=dim_cnn,
                dim_output_cnn=space.shape[0],
                act_fun=act_fun,
                weight_init=weight_init,
                bias_init=bias_init,
                cnn_norm=cnn_norm,
                dist_weight_init=dist_weight_init,
                dist_bias_init=dist_bias_init,
                image_size=image_size,
                **filtered_kwargs
            )

    def forward(self, inputs):
        """Decodes stochastic features into multiple observation predictions."""
        return {
            name: decoder(inputs) for name, decoder in self.decoders.items()
        }
