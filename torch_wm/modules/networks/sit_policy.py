import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch_wm.modules.blocks.multi_layer_perceptron import MultiLayerPerceptron
from torch_wm.core.registry import ModuleRegistry
from torch_wm.modules.networks.diffusion_policy import DiffusionDistWrapper


@ModuleRegistry.register('sit')
class SiTPolicyNetwork(nn.Module):
    """
    Scalable Interpolant (Flow Matching / Rectified Flow) Policy Network.
    Uses Continuous Normalizing Flows (ODE) instead of SDE (Diffusion).

    Interface contract with TSSM.imagine():
        - __call__(feat) must return an object with .rsample() → action tensor
        - compute_loss(feat, action) must return per-sample MSE loss
        - sample(feat) must return DiffusionDistWrapper
    """

    def __init__(
            self,
            num_actions,
            hidden_size=512,
            num_mlp_layers=5,
            feat_size=32*32+512,
            discrete=True,
            n_timesteps=5,
            **kwargs
        ):
        super().__init__()
        self.num_actions = num_actions
        self.discrete = discrete
        self.n_timesteps = n_timesteps

        # Dimensions: Continuous action space size (or num logits if discrete)
        self.action_dim = num_actions

        # Vector Field Prediction Network: Predicts velocity v_t given (x_t, state_feat, t)
        self.time_dim = 64
        self.mlp = MultiLayerPerceptron(
            dim_input=feat_size + self.action_dim + self.time_dim,
            dim_layers=[hidden_size] * num_mlp_layers + [self.action_dim],
            act_fun=[nn.SiLU] * num_mlp_layers + [None],
            bias=True
        )

    def get_time_embedding(self, t):
        """Sinusoidal time embedding for t in [0, 1]."""
        t_scaled = t * 1000.0
        half_dim = self.time_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t_scaled.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

    def predict_velocity(self, state_feat, x_t, t):
        """
        Predicts the velocity vector (vector field) at x_t and time t.
        state_feat: (B, FeatDim) or (B, L, FeatDim)
        x_t: (B, ActDim) or (B, L, ActDim)
        t: (B,) in [0, 1]
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
        v_pred = self.mlp(x)

        if reshape_back:
            v_pred = v_pred.reshape(B, L, -1)

        return v_pred

    def forward(self, state_feat):
        """
        Called by TSSM.imagine() as p_net(feat).
        Must return an object with .rsample() to be compatible with imagine().
        Runs ODE sampling internally, returns DiffusionDistWrapper.
        """
        return self._sample_internal(state_feat)

    def _sample_internal(self, state_feat):
        """Generate actions via ODE Euler Integration (Flow Matching)."""
        shape = state_feat.shape[:-1] + (self.action_dim,)
        device = state_feat.device

        # Start from base noise distribution x_0 at t=0
        x = torch.randn(shape, device=device)

        dt = 1.0 / self.n_timesteps

        # Euler integration from t=0 to t=1
        for i in range(self.n_timesteps):
            t_val = i * dt
            t = torch.full(shape[:-1], t_val, device=device, dtype=torch.float32)

            # Predict velocity
            v = self.predict_velocity(state_feat, x, t)

            # Euler step (fully differentiable! No detach so gradients flow through all t)
            x = x + v * dt

        return SiTDistWrapper(x, discrete=self.discrete)

    def compute_loss(self, state_feat, action):
        """
        Compute the Flow Matching (Rectified Flow) MSE loss.
        x_0 ~ N(0, I)
        x_1 = action (target)
        x_t = t * x_1 + (1 - t) * x_0  (linear interpolant)
        Target velocity v_t = x_1 - x_0
        """
        B = action.shape[0]
        device = action.device

        # Sample random continuous time t in [0, 1]
        t = torch.rand((B,), device=device)

        # Base noise distribution x_0
        x_0 = torch.randn_like(action)
        x_1 = action

        # Reshape t for broadcasting
        t_view = t.view(-1, 1)

        # Interpolant (Straight line from x_0 to x_1)
        x_t = t_view * x_1 + (1.0 - t_view) * x_0

        # Target velocity is constant along each path: x_1 - x_0
        target_v = x_1 - x_0

        # Predict velocity
        pred_v = self.predict_velocity(state_feat, x_t, t)

        # Flow Matching Loss: MSE between predicted velocity and target velocity
        loss = F.mse_loss(pred_v, target_v, reduction='none')
        # Mean over action dimension, keep batch dimension
        loss = loss.mean(dim=-1)
        return loss

    @torch.no_grad()
    def sample(self, state_feat):
        """Generate actions via ODE (inference only, no gradient)."""
        return self._sample_internal(state_feat)


class SiTDistWrapper:
    """Wraps SiT ODE output logits into a proper OneHotCategoricalStraightThrough distribution.

    Mirrors TWISTER's OneHotDist (one_hot_dist.py):
        class OneHotDist(OneHotCategoricalStraightThrough):
            - rsample() returns one-hot forward, probs gradient backward (STE)
            - log_prob() returns valid categorical log probabilities
            - entropy() returns categorical entropy

    This is REQUIRED for REINFORCE actor gradient (actor_grad = "reinforce").
    """

    def __init__(self, x, discrete=True, uniform_mix=0.01):
        self.x = x  # raw ODE output logits
        self.discrete = discrete

        if self.discrete:
            # Build a proper OneHotCategoricalStraightThrough distribution
            # exactly like TWISTER's OneHotDist.__init__
            logits = self.x
            if uniform_mix > 0:
                probs = F.softmax(logits, dim=-1)
                probs = (1 - uniform_mix) * probs + uniform_mix / probs.shape[-1]
                logits = torch.log(probs)
            self._dist = torch.distributions.OneHotCategoricalStraightThrough(logits=logits)
        else:
            self._dist = None

    def rsample(self):
        """Reparameterized sample — OneHotCategoricalStraightThrough.rsample().

        Forward: returns one-hot vector (discrete action for World Model).
        Backward: gradients flow through probs (Straight-Through Estimator).
        Exactly matches TWISTER's TSSM.imagine() line 179:
            policy = lambda s: p_net(self.get_feat(s).detach()).rsample()
        """
        if self.discrete:
            return self._dist.rsample()
        return self.x

    def sample(self):
        """Stochastic sample — used for environment interaction."""
        if self.discrete:
            return self._dist.sample()
        return self.x

    def mode(self):
        """Deterministic greedy action — used for evaluation.

        Matches TWISTER's OneHotDist.mode():
            mode = super().mode
            return mode.detach() + (self.logits - self.logits.detach())
        """
        if self.discrete:
            _mode = self._dist.mode
            return _mode.detach() + (self._dist.logits - self._dist.logits.detach())
        return self.x

    def entropy(self):
        """Categorical entropy from the distribution.

        Used in REINFORCE actor loss (TWISTER line 1062):
            policy_ent = policy_dist.entropy()[:, :-1]
        """
        if self.discrete:
            return self._dist.entropy()
        return torch.zeros(self.x.shape[:-1], device=self.x.device)

    def log_prob(self, action):
        """Categorical log probability.

        Used in REINFORCE actor loss (TWISTER line 1057):
            actor_loss = policy_dist.log_prob(img_states["action"].detach())[:, :-1] * advantage.detach()
        """
        if self.discrete:
            return self._dist.log_prob(action)
        return torch.zeros(self.x.shape[:-1], device=self.x.device)
