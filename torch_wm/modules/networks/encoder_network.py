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
from torch_wm.utils.checkpoint import checkpoint_forward

class EncoderNetwork(nn.Module):

    def __init__(
        self,
        dim_input_cnn=3,
        dim_cnn=32,
        act_fun=nn.SiLU,
        weight_init="dreamerv3_normal",
        bias_init="zeros",
        cnn_norm={"class": "LayerNorm", "params": {"eps": 1e-3}},
        image_size=(64, 64),
        stoch_size=32,
        discrete=32,
        dist_weight_init="xavier_uniform",
        dist_bias_init="zeros",
        uniform_mix=0.01,
        use_checkpointing=False,
        **kwargs
    ):
        super(EncoderNetwork, self).__init__()

        # Params
        self.dim_input_cnn = dim_input_cnn
        self.dim_cnn = dim_cnn
        self.image_size = image_size
        self.dim_concat = 4*4*8*dim_cnn
        self.stoch_size = stoch_size
        self.discrete = discrete
        self.uniform_mix = uniform_mix
        self.use_checkpointing = use_checkpointing

        # Architecture configuration (configurable via YAML)
        self.dim_layers = kwargs.get("encoder_dim_layers", [dim_cnn, 2*dim_cnn, 4*dim_cnn, 8*dim_cnn])
        self.kernel_size = kwargs.get("kernel_size", 4)
        self.strides = kwargs.get("strides", 2)

        # CNN layers
        pretrained_config = kwargs.get("pretrained", {})
        if isinstance(pretrained_config, dict) and pretrained_config.get("enabled", False):
            import torchvision.models as tv_models
            from carla_env.toolkit.utils import get_logger
            log = get_logger(log_dir=".", job_name="encoder_network")

            backbone_name = pretrained_config.get("backbone", "resnet18")
            weights = pretrained_config.get("weights", "IMAGENET1K_V1")

            log.info(f"[Registry] Loading Pretrained Backbone: {backbone_name} with weights {weights}")

            try:
                # Instantiate torchvision model
                backbone = getattr(tv_models, backbone_name)(weights=weights)

                # Add ImageNet standardization wrapper to bridge [-0.5, 0.5] inputs!
                class ImageNetWrapper(nn.Module):
                    def __init__(self, core_net):
                        super().__init__()
                        self.core_net = core_net
                        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
                        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
                    def forward(self, x):
                        # Convert from Dreamer [-0.5, 0.5] to [0, 1]
                        x = x + 0.5
                        # Apply ImageNet normalization
                        x = (x - self.mean) / self.std
                        return self.core_net(x)

                if hasattr(backbone, 'fc'):
                    core_net = nn.Sequential(*list(backbone.children())[:-2])
                elif hasattr(backbone, 'classifier'):
                    core_net = nn.Sequential(*list(backbone.children())[:-2])
                else:
                    core_net = backbone

                self.cnn = ImageNetWrapper(core_net)

                # Freeze weights if requested
                if pretrained_config.get("freeze", False):
                    for param in self.cnn.parameters():
                        param.requires_grad = False
                    log.info(f"[Registry] Frozen Pretrained Backbone: {backbone_name}")

            except Exception as e:
                log.error(f"Failed to load pretrained backbone {backbone_name}: {e}. Falling back to default CNN.")
                self.cnn = modules.ConvNeuralNetwork(
                    dim_input=self.dim_input_cnn,
                    dim_layers=self.dim_layers,
                    kernel_size=self.kernel_size,
                    strides=self.strides,
                    act_fun=act_fun,
                    padding=kwargs.get("padding", "same"),
                    weight_init=weight_init,
                    bias_init=bias_init,
                    norm=cnn_norm,
                    channels_last=False,
                    bias=cnn_norm is None,
                    blocks=kwargs.get("cnn_blocks", 0)
                )
        else:
            self.cnn = modules.ConvNeuralNetwork(
                dim_input=self.dim_input_cnn,
                dim_layers=self.dim_layers,
                kernel_size=self.kernel_size,
                strides=self.strides,
                act_fun=act_fun,
                padding=kwargs.get("padding", "same"),
                weight_init=weight_init,
                bias_init=bias_init,
                norm=cnn_norm,
                channels_last=False,
                bias=cnn_norm is None,
                blocks=kwargs.get("cnn_blocks", 0)
            )

        # Dynamic Shape Inference (removes hardcoding like 4*4 or 8*8)
        self.dim_concat = self._infer_cnn_output_shape()

        self.is_standalone = kwargs.get("is_standalone", True)
        if self.is_standalone:
            # Representation Network: concat features -> logits
            self.representation_network = modules.Linear(
                in_features=self.dim_concat,
                out_features=discrete * stoch_size if discrete else 2 * stoch_size,
                weight_init=dist_weight_init,
                bias_init=dist_bias_init,
            )

    def _infer_cnn_output_shape(self):
        """Run a dummy forward pass to determine the CNN output dimension."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.dim_input_cnn, self.image_size[0], self.image_size[1])
            dummy_output = self.cnn(dummy_input)
            return dummy_output.reshape(1, -1).size(1)

    def get_dist(self, state):

        return torch.distributions.Independent(distributions.OneHotDist(logits=state['logits'], uniform_mix=self.uniform_mix), 1)

    def forward_cnn(self, x):
        # ROOT-TO-TIP FIX: Force casting and scaling for ByteTensors inside the model
        if x.dtype == torch.uint8:
            x = x.float() / 255.0 - 0.5

        # Ensure same device and dtype as weights
        weight_ptr = next(self.parameters())
        x = x.to(device=weight_ptr.device, dtype=weight_ptr.dtype)

        shape = x.shape
        x = x.reshape((-1,) + shape[-3:])
        x = checkpoint_forward(self.cnn, x, use_checkpointing=self.use_checkpointing, training=self.training)
        x = x.reshape(shape[:-3] + (self.dim_concat,))
        return x

    def forward(self, inputs):
        if not self.is_standalone:
            raise RuntimeError("Cannot call forward() on EncoderNetwork when is_standalone=False. Call forward_cnn() instead or initialize with is_standalone=True.")

        outputs = self.forward_cnn(inputs)

        # Dist params
        dist_params = {'logits': self.representation_network(outputs).reshape(outputs.shape[:-1] + (self.stoch_size, self.discrete))}

        # Sample
        stoch = self.get_dist(dist_params).rsample()

        # Return State
        return {"stoch": stoch, "latent": outputs, **dist_params}

class MultiEncoderNetwork(nn.Module):
    """
    Modular Encoder for multi-sensor observations (Camera, Birdeye, etc.)
    Dynamically creates branches based on the YAML configuration.

    Branches with tssm=True (default) contribute to the world model's stochastic state.
    Branches with tssm=False (e.g. BEV) are encoded separately as conditioning signals
    for the policy network, without entering the TSSM dynamics.
    """
    def __init__(
        self,
        obs_config,
        obs_space,
        dim_cnn=32,
        act_fun=nn.SiLU,
        weight_init="dreamerv3_normal",
        bias_init="zeros",
        cnn_norm={"class": "LayerNorm", "params": {"eps": 1e-3}},
        stoch_size=32,
        discrete=32,
        dist_weight_init="xavier_uniform",
        dist_bias_init="zeros",
        uniform_mix=0.01,
        use_checkpointing=False,
        **kwargs
    ):
        super(MultiEncoderNetwork, self).__init__()

        self.stoch_size = stoch_size
        self.discrete = discrete
        self.uniform_mix = uniform_mix
        self.encoders = nn.ModuleDict()
        self.dim_concat = 0           # TSSM branch feature dim
        self.dim_signal = 0           # Signal-only branch feature dim
        self._tssm_branches = []      # Branch names feeding world model
        self._signal_branches = []    # Branch names for policy conditioning only

        # Dynamically create encoders for each enabled sensor
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
            print(f"[DEBUG Encoder] Checking {name}, shape={space.shape}")
            if len(space.shape) < 3:
                continue

            # Allow fallback if no specific config exists
            config = obs_config.get(name, AttrDict())
            if not isinstance(config, AttrDict):
                config = AttrDict(config)

            # Filter kwargs to avoid multiple values for image_size
            sub_kwargs = {k: v for k, v in kwargs.items() if k != "image_size"}

            # Allow sensor config to override global kwargs
            sensor_kwargs = sub_kwargs.copy()
            for k, v in config.items():
                if k not in ['shape', 'attributes', 'tssm', 'decode']:
                    sensor_kwargs[k] = v

            image_size = (space.shape[1], space.shape[2])

            encoder = EncoderNetwork(
                dim_input_cnn=space.shape[0],
                dim_cnn=dim_cnn,
                act_fun=act_fun,
                weight_init=weight_init,
                bias_init=bias_init,
                cnn_norm=cnn_norm,
                image_size=image_size,
                stoch_size=stoch_size,
                discrete=discrete,
                is_standalone=False,
                **sensor_kwargs
            )
            self.encoders[name] = encoder

            # tssm=True (default): feeds world model. tssm=False: signal only.
            is_tssm = config.get("tssm", True)
            if is_tssm:
                self._tssm_branches.append(name)
                self.dim_concat += encoder.dim_concat
            else:
                self._signal_branches.append(name)
                self.dim_signal += encoder.dim_concat

        # Representation network maps TSSM features → stoch space
        self.representation_network = modules.Linear(
            in_features=self.dim_concat,
            out_features=discrete * stoch_size if discrete else 2 * stoch_size,
            weight_init=dist_weight_init,
            bias_init=dist_bias_init,
        )

    def get_dist(self, state):
        return torch.distributions.Independent(distributions.OneHotDist(logits=state['logits'], uniform_mix=self.uniform_mix), 1)

    def forward_cnn(self, inputs):
        """Encodes TSSM modalities and concatenates their features (backward compat)."""
        features = []
        for name in self._tssm_branches:
            if name in inputs:
                features.append(self.encoders[name].forward_cnn(inputs[name]))

        if not features:
            raise ValueError(f"No TSSM sensor data found! Expected: {self._tssm_branches}, got: {list(inputs.keys())}")

        return torch.cat(features, dim=-1)

    def forward(self, inputs):
        # ── TSSM branches → stoch (world model state) ──
        tssm_cat = self.forward_cnn(inputs)

        first_name = self._tssm_branches[0]
        shape_head = inputs[first_name].shape[:-3]

        dist_params = {'logits': self.representation_network(tssm_cat).reshape(shape_head + (self.stoch_size, self.discrete))}
        stoch = self.get_dist(dist_params).rsample()

        result = {"stoch": stoch, "latent": tssm_cat, **dist_params}

        # ── Signal branches → separate features (policy conditioning) ──
        if self._signal_branches:
            signal_parts = []
            for name in self._signal_branches:
                if name in inputs:
                    signal_parts.append(self.encoders[name].forward_cnn(inputs[name]))
            if signal_parts:
                result["signal"] = torch.cat(signal_parts, dim=-1)

        return result
