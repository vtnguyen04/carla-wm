import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch_wm.modules.blocks.multi_layer_perceptron import MultiLayerPerceptron
from torch_wm.core.registry import ModuleRegistry


@ModuleRegistry.register('diffusion')
class DiffusionPolicyNetwork(nn.Module):
    """
    DDPM (Denoising Diffusion Probabilistic Model) Policy Network.
    Diffuses a continuous representation of actions (e.g. logits for discrete spaces).

    Interface contract with TSSM.imagine():
        - __call__(feat) must return an object with .rsample() → action tensor
        - compute_loss(feat, action) must return per-sample MSE loss
    """

    def __init__(
            self,
            num_actions,
            hidden_size=512,
            num_mlp_layers=5,
            feat_size=32*32+512,
            discrete=True,
            n_timesteps=10,
            beta_start=1e-4,
            beta_end=0.02,
            **kwargs
        ):
        super().__init__()
        self.num_actions = num_actions
        self.discrete = discrete
        self.n_timesteps = n_timesteps

        # Dimensions: Continuous action space size (or num logits if discrete)
        self.action_dim = num_actions

        # Noise Prediction Network: Predicts epsilon given (x_t, state_feat, t)
        self.time_dim = 64
        self.mlp = MultiLayerPerceptron(
            dim_input=feat_size + self.action_dim + self.time_dim,
            dim_layers=[hidden_size] * num_mlp_layers + [self.action_dim],
            act_fun=nn.SiLU,
            bias=True
        )

        # DDPM schedule parameters
        betas = torch.linspace(beta_start, beta_end, n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def get_time_embedding(self, t):
        """Sinusoidal time embedding."""
        half_dim = self.time_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

    def predict_noise(self, state_feat, x_t, t):
        """
        Predicts the noise added to x_t at timestep t.
        state_feat: (B, FeatDim) or (B, L, FeatDim)
        x_t: (B, ActDim) or (B, L, ActDim)
        t: (B,)
        """
        t_emb = self.get_time_embedding(t)
        reshape_back = False
        if state_feat.dim() == 3:
            reshape_back = True
            B, L, F_dim = state_feat.shape
            state_feat = state_feat.reshape(B*L, F_dim)
            x_t = x_t.reshape(B*L, -1)
            t_emb = t_emb.reshape(B*L, -1)

        x = torch.cat([state_feat, x_t, t_emb], dim=-1)
        noise_pred = self.mlp(x)

        if reshape_back:
            noise_pred = noise_pred.reshape(B, L, -1)

        return noise_pred

    def forward(self, state_feat):
        """
        Called by TSSM.imagine() as p_net(feat).
        Must return an object with .rsample() to be compatible with imagine().
        """
        return self._sample_internal(state_feat)

    def _sample_internal(self, state_feat):
        """Generate actions via reverse diffusion process."""
        shape = state_feat.shape[:-1] + (self.action_dim,)
        device = state_feat.device

        # Start from pure noise
        x = torch.randn(shape, device=device)

        for i in reversed(range(self.n_timesteps)):
            t = torch.full(shape[:-1], i, device=device, dtype=torch.float32)

            # Predict noise
            noise_pred = self.predict_noise(state_feat, x, t)

            # Reverse step equations
            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            beta = self.betas[i]

            # Compute x_{t-1}
            coef = beta / torch.sqrt(1.0 - alpha_cumprod)
            mean = (1.0 / torch.sqrt(alpha)) * (x - coef * noise_pred)

            if i > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta)
                x = mean + sigma * noise
            else:
                x = mean

        # Import SiTDistWrapper (stochastic) instead of old DiffusionDistWrapper
        from torch_wm.modules.networks.sit_policy import SiTDistWrapper
        return SiTDistWrapper(x, discrete=self.discrete)

    def compute_loss(self, state_feat, action):
        """Compute the DDPM MSE loss for training."""
        device = action.device

        # Sample random timesteps
        t_shape = action.shape[:-1]
        t = torch.randint(0, self.n_timesteps, t_shape, device=device).float()

        # Add noise to true actions
        noise = torch.randn_like(action)

        # Reshape for broadcasting
        t_idx = t.long()
        broadcast_shape = t_shape + (1,)
        sqrt_alpha = self.sqrt_alphas_cumprod[t_idx].view(broadcast_shape)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t_idx].view(broadcast_shape)

        # x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * epsilon
        x_t = sqrt_alpha * action + sqrt_one_minus * noise

        # Predict noise
        noise_pred = self.predict_noise(state_feat, x_t, t)

        # Loss is MSE between predicted noise and actual noise
        loss = F.mse_loss(noise_pred, noise, reduction='none')
        loss = loss.mean(dim=-1)  # Mean over action dim, keep batch dim
        return loss

    @torch.no_grad()
    def sample(self, state_feat):
        """Generate actions (inference only, no gradient)."""
        return self._sample_internal(state_feat)


class DiffusionDistWrapper:
    """Legacy wrapper — kept for backward compatibility.
    New code should use SiTDistWrapper from sit_policy.py which has proper
    stochastic sampling and .rsample() support.
    """
    def __init__(self, x, discrete=True):
        self.x = x
        self.discrete = discrete

    def rsample(self):
        return self.sample()

    def sample(self):
        if self.discrete:
            idx = self.x.argmax(dim=-1, keepdim=True)
            one_hot = torch.zeros_like(self.x)
            one_hot.scatter_(-1, idx, 1.0)
            return one_hot
        return self.x

    def mode(self):
        return self.sample()

    def entropy(self):
        return torch.zeros(self.x.shape[:-1], device=self.x.device)

    def log_prob(self, action):
        return torch.zeros(self.x.shape[:-1], device=self.x.device)
