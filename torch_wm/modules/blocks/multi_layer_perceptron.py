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
import torch.nn.functional as F

# NeuralNets
from torch_wm import modules
from torch_wm.utils import get_module_and_params
from torch_wm.utils.checkpoint import checkpoint_forward

class MultiLayerPerceptron(nn.Module):

    def __init__(self, dim_input, dim_layers, act_fun="ReLU", norm=None, drop_rate=0.0, weight_init="default", bias_init="default", bias=True, residual=False, use_checkpointing=False):
        super(MultiLayerPerceptron, self).__init__()

        # Get act_fun and norm
        act_fun, act_fun_params = get_module_and_params(act_fun, modules.act_dict)
        norm, norm_params = get_module_and_params(norm, modules.norm_dict)
            
        # Single Layer
        if isinstance(dim_layers, int):
            dim_layers = [dim_layers]

        # MLP Layers
        self.layers = nn.ModuleList([nn.Sequential(

            # Linear
            modules.Linear(
                in_features=dim_input if layer_id == 0 else dim_layers[layer_id - 1], 
                out_features=dim_layers[layer_id], 
                weight_init=weight_init[layer_id] if isinstance(weight_init, list) else weight_init, 
                bias_init=bias_init[layer_id] if isinstance(bias_init, list) else bias_init, 
                bias=bias[layer_id] if isinstance(bias, list) else bias
            ),
            
            # Norm
            (norm[layer_id](dim_layers[layer_id], **(norm_params[layer_id] if isinstance(norm_params, list) else norm_params), channels_last=True) if norm[layer_id] is not None else nn.Identity()) if isinstance(norm, list) else (norm(dim_layers[layer_id], **norm_params, channels_last=True) if norm is not None else nn.Identity()),
            
            # Act fun
            (act_fun[layer_id](**(act_fun_params[layer_id] if isinstance(act_fun_params, list) else act_fun_params)) if act_fun[layer_id] is not None else nn.Identity()) if isinstance(act_fun, list) else (act_fun(**act_fun_params) if act_fun is not None else nn.Identity()),
            
            # Dropout
            nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity() 
        ) for layer_id in range(len(dim_layers))])

        # Residual
        if isinstance(residual, list):
            assert len(residual) == len(self.layers)
            self.residual = residual
        else:
            self.residual = [residual for layer_id in range(len(self.layers))]

        self.use_checkpointing = use_checkpointing

    def forward(self, x):

        # Layers
        for layer_id, layer in enumerate(self.layers):
            x = checkpoint_forward(
                layer, 
                x, 
                use_checkpointing=self.use_checkpointing, 
                training=self.training, 
                residual=self.residual[layer_id]
            )

        return x