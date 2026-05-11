import os
import torch
import torch.nn as nn
from torch_wm.core.registry import ModuleRegistry
from torch_wm.modules.networks.vit import vit_large

from torch_wm import distributions
from torch_wm import modules

@ModuleRegistry.register('vjepa_encoder')
class VJEPAEncoderNetwork(nn.Module):
    """
    V-JEPA 2.1 Pretrained Encoder wrapper for the Unified World Model.
    """

    def __init__(
        self,
        checkpoint_path="models/vjepa2_1_vitl_dist_vitG_384.pt",
        patch_size=16,
        tubelet_size=2,
        freeze=True,
        image_size=(384, 384),
        stoch_size=32,
        discrete=32,
        uniform_mix=0.01,
        **kwargs
    ):
        super().__init__()

        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.image_size = image_size
        self.stoch_size = stoch_size
        self.discrete = discrete
        self.uniform_mix = uniform_mix

        print(f"[V-JEPA Encoder] Initializing ViT-Large with RoPE, patch_size={patch_size}")

        self.model = vit_large(
            patch_size=patch_size,
            img_size=image_size,
            num_frames=tubelet_size,
            tubelet_size=tubelet_size,
            use_sdpa=True,
            use_SiLU=False,
            wide_SiLU=True,
            uniform_power=False,
            use_rope=True,
            img_temporal_dim_size=1,
            interpolate_rope=True,
        )

        if os.path.exists(checkpoint_path):
            print(f"[V-JEPA Encoder] Loading weights from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            state_dict = checkpoint["ema_encoder"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=True)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1))

        h_patches = image_size[0] // patch_size
        w_patches = image_size[1] // patch_size
        num_tokens = h_patches * w_patches
        self.dim_concat = num_tokens * self.model.embed_dim

        # ADD REPRESENTATION NETWORK (Matches EncoderNetwork interface)
        trainable = kwargs.get("trainable", True)
        if trainable:
            self.representation_network = modules.Linear(
                in_features=self.dim_concat,
                out_features=discrete * stoch_size if discrete else 2 * stoch_size,
            )
        else:
            self.representation_network = None

    def get_dist(self, state):
        return torch.distributions.Independent(distributions.OneHotDist(logits=state['logits'], uniform_mix=self.uniform_mix), 1)

    def forward_cnn(self, x):
        if isinstance(x, dict):
            x = x.get("camera", next(iter(x.values())))

        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        # Standardize to [-0.5, 0.5] if needed, but here we expect [0, 1] for ImageNet
        # If input was already [-0.5, 0.5], shift to [0, 1]
        if x.min() < 0:
            x = x + 0.5

        if x.ndim == 4:
            x = x.unsqueeze(2).repeat(1, 1, self.tubelet_size, 1, 1)
        elif x.ndim == 5:
            # [B, T, C, H, W] -> [B, C, T, H, W]
            x = x.permute(0, 2, 1, 3, 4)
            if x.shape[2] < self.tubelet_size:
                x = x.repeat(1, 1, self.tubelet_size // x.shape[2] + 1, 1, 1)[:, :, :self.tubelet_size]

        x = (x - self.mean) / self.std
        features = self.model(x)
        return features.reshape(features.shape[0], -1)

    def forward(self, inputs):
        # inputs: [B, T, C, H, W] or dict
        if isinstance(inputs, dict):
            x = inputs.get("camera", next(iter(inputs.values())))
        else:
            x = inputs

        shape = x.shape
        # Flatten B, T for CNN forward
        x_flat = x.reshape((-1,) + shape[-3:])
        latent = self.forward_cnn(x_flat)

        # Compute stoch/logits
        logits = self.representation_network(latent).reshape(shape[:-3] + (self.stoch_size, self.discrete))
        dist_params = {'logits': logits}
        stoch = self.get_dist(dist_params).rsample()

        # Reshape latent back to sequence
        latent_seq = latent.reshape(shape[:-3] + (self.dim_concat,))

        return {"stoch": stoch, "latent": latent_seq, **dist_params}
