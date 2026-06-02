import torch
import torch.nn as nn
from torch_wm import modules
from torch_wm import distributions

class DenseHead(nn.Module):
    """
    A unified dense network head that applies an MLP followed by a linear projection,
    and returns a specific distribution.
    Replaces repetitive RewardNetwork, ValueNetwork, and ContinueNetwork.
    """
    def __init__(
        self, 
        dist_type="symlog",
        bins=255,
        out_dim=1,
        hidden_size=512, 
        act_fun=nn.SiLU, 
        num_mlp_layers=2, 
        feat_size=32*32+512, 
        weight_init="dreamerv3_normal", 
        bias_init="zeros", 
        norm={"class": "LayerNorm", "params": {"eps": 1e-3}}, 
        dist_weight_init="zeros", 
        dist_bias_init="zeros"
    ):
        super(DenseHead, self).__init__()
        self.dist_type = dist_type
        
        # Determine output projection size
        if self.dist_type == "symlog":
            proj_size = bins
        elif self.dist_type == "binary":
            proj_size = out_dim
        elif self.dist_type == "normal":
            proj_size = out_dim * 2 # mean and std
        else:
            proj_size = out_dim

        self.mlp = modules.MultiLayerPerceptron(
            dim_input=feat_size, 
            dim_layers=[hidden_size for _ in range(num_mlp_layers)], 
            act_fun=act_fun, 
            weight_init=weight_init, 
            bias_init=bias_init, 
            norm=norm, 
            bias=norm is None
        )
        self.linear_proj = modules.Linear(
            hidden_size, 
            proj_size, 
            weight_init=dist_weight_init, 
            bias_init=dist_bias_init
        )

    def forward(self, x):
        # Apply MLP and Output Projection
        x = self.mlp(x)
        logits = self.linear_proj(x)

        # Wrap in Distribution
        if self.dist_type == "symlog":
            return distributions.SymLogDiscreteDist(logits=logits, reinterpreted_batch_ndims=1, low=-20, high=20)
        elif self.dist_type == "binary":
            return torch.distributions.Independent(distributions.Bernoulli(logits=logits), 1)
        elif self.dist_type == "normal":
            import torch.nn.functional as F
            mean, std = torch.chunk(logits, chunks=2, dim=-1)
            # Default normal head bounds (as in policy network but simplified for general use)
            mean = F.tanh(mean)
            std = 0.9 * F.sigmoid(std + 2.0) + 0.1 # [0.1, 1.0]
            return torch.distributions.Independent(distributions.Normal(mean, std), 1)
        else:
            raise ValueError(f"Unknown dist_type: {self.dist_type}")
