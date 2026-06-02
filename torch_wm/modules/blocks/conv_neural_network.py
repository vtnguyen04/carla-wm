# Copyright 2025, Maxime Burchi.
# Modifications copyright 2026, Vo Thanh Nguyen.
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
from torch_wm.utils import get_module_and_params

class ConvNeuralNetwork(nn.Module):

    def __init__(self, dim_input, dim_layers, kernel_size, strides=1, norm=None, act_fun="ReLU", drop_rate=0.0, padding="same", dim=2, channels_last=False, residual=False, weight_init="default", bias_init="default", bias=True, blocks=0):
        super(ConvNeuralNetwork, self).__init__()

        # Convs
        conv = {
            2: modules.Conv2d
        }

        # Get act_fun and norm
        act_fun, act_fun_params = get_module_and_params(act_fun, modules.act_dict)
        norm, norm_params = get_module_and_params(norm, modules.norm_dict)

        # Params
        self.strides = strides
        self.residual = residual

        # Single Layer
        if isinstance(dim_layers, int):
            dim_layers = [dim_layers]

        # CNN Layers
        self.layers = nn.ModuleList()
        for layer_id in range(len(dim_layers)):
            in_d = dim_input if layer_id == 0 else dim_layers[layer_id - 1]
            out_d = dim_layers[layer_id]
            
            curr_norm = norm[layer_id] if isinstance(norm, list) else norm
            curr_norm_params = norm_params[layer_id] if isinstance(norm_params, list) else norm_params
            curr_act = act_fun[layer_id] if isinstance(act_fun, list) else act_fun
            curr_act_params = act_fun_params[layer_id] if isinstance(act_fun_params, list) else act_fun_params
            
            main_conv = nn.Sequential(
                conv[dim](
                    in_d, 
                    out_d, 
                    kernel_size[layer_id] if isinstance(kernel_size, list) else kernel_size, 
                    stride=strides[layer_id] if isinstance(strides, list) else strides, 
                    padding=padding[layer_id] if isinstance(padding, list) else padding, 
                    channels_last=channels_last, 
                    weight_init=weight_init[layer_id] if isinstance(weight_init, list) else weight_init, 
                    bias_init=bias_init[layer_id] if isinstance(bias_init, list) else bias_init, 
                    bias=bias[layer_id] if isinstance(bias, list) else bias
                ), 
                (curr_norm(out_d, **curr_norm_params, channels_last=channels_last) if curr_norm is not None else nn.Identity()),
                (curr_act(**curr_act_params) if curr_act is not None else nn.Identity()),
                nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity() 
            )
            
            if blocks > 0:
                stage_blocks = nn.ModuleList()
                for _ in range(blocks):
                    b = nn.Sequential(
                        conv[dim](out_d, out_d, 3, padding="same", channels_last=channels_last, weight_init=weight_init[layer_id] if isinstance(weight_init, list) else weight_init, bias_init=bias_init[layer_id] if isinstance(bias_init, list) else bias_init, bias=bias[layer_id] if isinstance(bias, list) else bias),
                        (curr_norm(out_d, **curr_norm_params, channels_last=channels_last) if curr_norm is not None else nn.Identity()),
                        (curr_act(**curr_act_params) if curr_act is not None else nn.Identity()),
                        conv[dim](out_d, out_d, 3, padding="same", channels_last=channels_last, weight_init=weight_init[layer_id] if isinstance(weight_init, list) else weight_init, bias_init=bias_init[layer_id] if isinstance(bias_init, list) else bias_init, bias=bias[layer_id] if isinstance(bias, list) else bias),
                        (curr_norm(out_d, **curr_norm_params, channels_last=channels_last) if curr_norm is not None else nn.Identity())
                    )
                    stage_blocks.append(nn.ModuleDict({'block': b, 'act': curr_act(**curr_act_params) if curr_act is not None else nn.Identity()}))
                self.layers.append(nn.ModuleDict({'main': main_conv, 'blocks': stage_blocks}))
            else:
                self.layers.append(main_conv)

    def forward(self, x, x_len=None, num_batch_axes=1):

        # Flatten batch axes
        if num_batch_axes > 1:
            shape = x.shape
            x = x.flatten(start_dim=0, end_dim=num_batch_axes-1)

        # Layers
        for layer in self.layers:

            # Forward
            if isinstance(layer, nn.ModuleDict):
                x = layer['main'](x)
                for stage_block in layer['blocks']:
                    skip = x
                    x = stage_block['block'](x)
                    x = stage_block['act'](x + skip)
            else:
                if self.residual:
                    x = x + layer(x)
                else:
                    x = layer(x)

            # Update Sequence Lengths
            if x_len is not None:
                x_len = torch.div(x_len - 1, 2, rounding_mode='floor') + 1 # to generalize

        # UnFlatten batch axes
        if num_batch_axes > 1:
            x = x.reshape(shape[:-3] + x.shape[1:])

        return x if x_len==None else (x, x_len)