import os

import torch
import torch.nn as nn
from torch_wm.core.registry import ModuleRegistry
from torch_wm.modules.networks.vit_predictor import vit_predictor
from torch_wm import distributions

@ModuleRegistry.register('vjepa_predictor')
class VJEPAActionPredictor(nn.Module):
    """
    Action-Conditioned V-JEPA 2.1 Predictor for Latent Dynamics.
    """

    def __init__(
        self,
        num_actions,
        checkpoint_path="models/vjepa2_1_vitl_dist_vitG_384.pt",
        embed_dim=1024,
        predictor_embed_dim=384,
        teacher_embed_dim=1664,
        patch_size=16,
        tubelet_size=2,
        image_size=(384, 384),
        freeze=True, # FREEZE BY DEFAULT
        uniform_mix=0.01,
        stoch_size=32,
        discrete=32,
        **kwargs
    ):
        super().__init__()

        self.num_actions = num_actions
        self.embed_dim = embed_dim
        self.uniform_mix = uniform_mix
        self.stoch_size = stoch_size
        self.discrete = discrete

        print(f"[V-JEPA Predictor] Initializing Predictor for {num_actions} actions")

        # 1. Instantiate V-JEPA 2.1 Predictor
        self.predictor = vit_predictor(
            img_size=image_size,
            patch_size=patch_size,
            use_mask_tokens=True,
            embed_dim=embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            teacher_embed_dim=teacher_embed_dim,
            num_frames=tubelet_size,
            tubelet_size=tubelet_size,
            depth=12,
            num_heads=12,
            num_mask_tokens=8,
            use_rope=True,
            uniform_power=False,
            use_sdpa=True,
            use_silu=False,
            wide_silu=True,
            n_output_distillation=1,
            return_all_tokens=True,
            img_temporal_dim_size=1,
        )

        # 2. Action Projection
        self.action_encoder = nn.Linear(num_actions, predictor_embed_dim)

        # 3. Load Pretrained Weights
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            state_dict = checkpoint["predictor"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = self.predictor.load_state_dict(state_dict, strict=False)
            print(f"[V-JEPA Predictor] Weights loaded (strict=False): {msg}")

        # 4. Freeze weights
        if freeze:
            for param in self.predictor.parameters():
                param.requires_grad = False
            print("[V-JEPA Predictor] Weights FROZEN.")

    def get_dist(self, state):
        """Returns the distribution over stochastic states."""
        return torch.distributions.Independent(distributions.OneHotDist(logits=state['logits'], uniform_mix=self.uniform_mix), 1)

    def get_feat(self, state):
        """
        Returns the flat feature vector for downstream heads.
        State is expected to be a dict containing 'stoch' and 'latent'.
        """
        stoch = state['stoch']
        latent = state['latent']
        stoch_flat = stoch.reshape(stoch.shape[:-2] + (-1,))
        return torch.cat([stoch_flat, latent], dim=-1)

    def observe(self, state_in, action, is_first=None):
        return state_in, state_in

    def imagine(self, state, action):
        """
        Predict the next state given current state and action.
        """
        # 1. Encode action
        action_emb = self.action_encoder(action) # [B, T, predictor_dim]

        # 2. Predict next latent (tokens)
        # Note: self.predictor usually requires specific tokens/masks.
        # We maintain the dummy return for this framework test to show logic flow.
        return state
