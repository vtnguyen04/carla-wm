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
from torch import nn
import torch.nn.functional as F
import numpy as np
try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

# torch_wm
from torch_wm.core.base_model import Model as BaseModel
from torch_wm.core.registry import ModuleRegistry
from torch_wm.modules import networks
from torch_wm.modules.networks import vjepa_encoder
from torch_wm.modules.dynamics import vjepa_dynamics
from torch_wm.structs import AttrDict
from torch_wm.optimizers import Adam
from torch_wm.schedulers.linear_warmup_scheduler import LinearWarmupCosineScheduler

# Standard library
import copy
import itertools
import numpy as np
import matplotlib.pyplot as plt
import io
from carla_env.toolkit.utils import get_logger

log = get_logger(log_dir=".", job_name="wm_agent")

class WMAgent(BaseModel):
    """
    Transformer-based World Model with Contrastive Representations (WMAgent)
    Optimized for Autonomous Driving in CARLA.
    """

    def __init__(self, env_name, override_config={}, name="WMAgent-CARLA", skip_env=False, obs_space=None):
        super(WMAgent, self).__init__(name=name)
        self._skip_env = skip_env

        # 1. Configuration & Modular Loading
        if isinstance(override_config, str) and (override_config.endswith(".yaml") or override_config.endswith(".yml")):
            from torch_wm.utils.yaml_config import load_yaml_config
            override_config = load_yaml_config(override_config)
        elif isinstance(override_config, str):
            import json
            override_config = json.loads(override_config)

        # Collate Function Integration
        from torch_wm.utils.collate_fn import CollateFn
        self.collate_fn = CollateFn(
            inputs_params=[{"axis": 0}, {"axis": 1}, {"axis": 2}, {"axis": 3}, {"axis": 4}],
            targets_params=[]
        )

        self.env_type = "carla"
        self.config = AttrDict()
        self.config.env_name = env_name

        # Merge architecture and base params directly from override_config
        self.config.update(override_config)

        # ── Type coercion for values YAML may parse as strings ──
        for lr_key in ("model_lr", "critic_lr", "actor_lr"):
            if hasattr(self.config, lr_key):
                setattr(self.config, lr_key, float(getattr(self.config, lr_key)))

        # Environment (optional — skipped for offline-only training)
        self.env = None
        self.env_eval = None

        # 4. Neural Networks Initialization
        # All network parameters are now strictly YAML-driven
        network_kwargs = dict(self.config)

        # Instantiate modular Encoder/Decoder using the observation config from YAML
        obs_config = self.config["env_params"]["observation"]
        if not isinstance(obs_config, AttrDict):
            obs_config = AttrDict(obs_config)

        # Use Registry for dynamic module loading
        encoder_type = self.config.get("encoder_type", "multi_encoder")
        if encoder_type == "multi_encoder":
            self.encoder_network = networks.MultiEncoderNetwork(
                obs_config=obs_config,
                obs_space=obs_space,
                **network_kwargs
            )
        else:
            encoder_cls = ModuleRegistry.get(encoder_type)
            self.encoder_network = encoder_cls(**network_kwargs)

        dynamics_type = self.config.get("dynamics_type", "tssm")
        if dynamics_type == "vjepa_predictor" and hasattr(self.encoder_network, "dim_concat"):
            feat_size = self.config.stoch_size * self.config.discrete + self.encoder_network.dim_concat if self.config.discrete else self.config.stoch_size + self.encoder_network.dim_concat
        else:
            feat_size = self.config.stoch_size * self.config.discrete + self.config.hidden_size if self.config.discrete else self.config.stoch_size + self.config.hidden_size

        # TWISTER parity: decoder receives ONLY stoch (not feats=stoch+deter)
        # This forces the encoder to encode all visual info into stoch,
        # preventing the decoder from bypassing stoch via deter (which kills entropy).
        decoder_feat_size = self.config.stoch_size * self.config.discrete if self.config.discrete else self.config.stoch_size
        if dynamics_type == "rssm":
            decoder_feat_size += self.config.hidden_size

        self.decoder_network = networks.MultiDecoderNetwork(
            obs_config=obs_config,
            obs_space=obs_space,
            feat_size=decoder_feat_size,
            encoder_network=self.encoder_network,
            **{k: v for k, v in network_kwargs.items() if k != 'feat_size'}
        )

        # Determine number of actions from env or config
        if self.env is not None:
            num_actions = self.env.num_actions
        else:
            # Look for num_actions in config, then env_params, then fallback to 15 (CARLA standard)
            num_actions = self.config.get("num_actions")
            if num_actions is None:
                num_actions = self.config.get("env_params", {}).get("action", {}).get("n_cmds", 15)
        self.num_actions = num_actions

        # ── Memory Optimization: Pass Gradient Checkpointing to TSSM ──
        use_checkpointing = self.config.get("gradient_checkpointing", False)
        dynamics_cls = ModuleRegistry.get(dynamics_type)

        if dynamics_type == "vjepa_predictor":
            self.dynamics_model = dynamics_cls(
                num_actions,
                **network_kwargs
            )
        else:
            self.dynamics_model = dynamics_cls(
                num_actions,
                stoch_size=self.config.stoch_size,
                discrete=self.config.discrete,
                hidden_size=self.config.hidden_size,
                num_blocks=self.config.num_blocks_trans,
                att_context_left=self.config.get("att_context_left", 8),
                use_checkpointing=use_checkpointing
            )
        # ── Head Networks (Using Nested Config if available) ──
        def get_head_kwargs(key, defaults, include_bins=True):
            cfg = self.config.get(key, {})
            kwargs = {
                "hidden_size": cfg.get("units", defaults["hidden_size"]),
                "num_mlp_layers": cfg.get("layers", defaults["num_mlp_layers"]),
            }
            if include_bins:
                kwargs["bins"] = cfg.get("bins", defaults.get("bins", 255))
            return kwargs


        policy_type = self.config.get("policy_type", "default_policy")
        policy_kwargs = get_head_kwargs("actor", {"hidden_size": self.config.hidden_size, "num_mlp_layers": 5}, include_bins=False)
        policy_kwargs.update({
            "feat_size": feat_size,
            "discrete": getattr(self.config, "discrete_actions", True),
        })

        # Inject additional parameters for specialized policies
        if policy_type in ["diffusion", "sit"]:
            policy_kwargs["n_timesteps"] = self.config.get("diffusion_steps", 10 if policy_type == "diffusion" else 5)
        elif policy_type == "default_policy":
            policy_kwargs["uniform_mix"] = getattr(self.config, "uniform_mix", 0.01)

        # Dynamic instantiation (GOF Factory Pattern via Registry)
        if policy_type == "default_policy":
            self.policy_network = networks.PolicyNetwork(num_actions, **policy_kwargs)
        else:
            policy_cls = ModuleRegistry.get(policy_type)
            self.policy_network = policy_cls(num_actions, **policy_kwargs)
        from torch_wm.modules.networks.dense_head import DenseHead
        self.value_network = DenseHead(
            dist_type="symlog",
            **get_head_kwargs("critic", {"hidden_size": self.config.hidden_size, "num_mlp_layers": 5}),
            feat_size=feat_size
        )
        self.reward_network = DenseHead(
            dist_type="symlog",
            **get_head_kwargs("reward_head", {"hidden_size": self.config.hidden_size, "num_mlp_layers": 5}),
            feat_size=feat_size
        )
        self.continue_network = DenseHead(
            dist_type="binary",
            **get_head_kwargs("cont_head", {"hidden_size": self.config.hidden_size, "num_mlp_layers": 5}, include_bins=False),
            feat_size=feat_size
        )

        # Option to define spaced-out CPC intervals instead of contiguous sequence
        self.contrastive_offsets = self.config.get("contrastive_offsets", [15])
        cpc_num_nets = len(self.contrastive_offsets) if self.contrastive_offsets else self.config.contrastive_steps

        cpc_enabled = self.config.get("modules", {}).get("losses", {}).get("cpc", {}).get("enabled", True)
        if cpc_enabled:
            self.contrastive_network = nn.ModuleList([
                networks.ContrastiveNetwork(
                    hidden_size=self.config.hidden_size,
                    out_size=self.config.stoch_size * self.config.discrete,
                    feat_size=feat_size,
                    embed_size=self.encoder_network.dim_concat
                ) for t in range(cpc_num_nets)
            ])
        else:
            self.contrastive_network = nn.ModuleList([])

        self.add_frozen("v_target", copy.deepcopy(self.value_network))
        self.register_buffer("perc_low", torch.tensor(0.0)); self.register_buffer("perc_high", torch.tensor(0.0))

        self._last_outputs = {}

        # Sub-models
        self.world_model = self.WorldModel(outer=self)

        # Factory Pattern for Actor Model
        actor_model_type = self.config.get("actor_model_type", "default_actor")
        if actor_model_type == "default_actor":
            self.actor_model = self.ActorModel(outer=self)
        else:
            actor_cls = ModuleRegistry.get(actor_model_type)
            self.actor_model = actor_cls(outer=self)

        self.critic_model = self.CriticModel(outer=self)

        # ── Interactive State (for Online ACT) ──
        self._current_state = None

    def act(self, obs, is_first, sample=True):
        """Perform a single interactive step in the CARLA environment."""
        obs = {k: torch.as_tensor(v).to(self.device) for k, v in obs.items()}
        is_first = torch.as_tensor(is_first).to(self.device).float()

        # 1. Preprocess
        obs = self.preprocess_inputs(obs, False)

        # 2. Reset if needed
        if is_first.any() or self._current_state is None:
            self._current_state = self.dynamics_model.initial(
                batch_size=obs[list(obs.keys())[0]].shape[0],
                device=self.device,
                dtype=torch.float32
            )
            self._prev_action = torch.zeros(
                obs[list(obs.keys())[0]].shape[0], 1, self.dynamics_model.num_actions,
                device=self.device
            )

        # 3. Step through World Model
        with torch.no_grad():
            latent = self.encoder_network(obs)
            # TSSM Observe expects (B, L, ...)
            # obs is (B, ...) from environment, needs unsqueeze(1) to add L=1
            latent_step = {k: v.unsqueeze(1) for k, v in latent.items()}

            post, _ = self.dynamics_model.observe(
                latent_step,
                self._prev_action,
                is_first.unsqueeze(1),
                prev_state=self._current_state
            )

            # ROOT-TO-TIP FIX: Slice hidden state to maintain constant attention window
            if post["hidden"] is not None:
                post["hidden"] = self.dynamics_model.slice_hidden(post["hidden"])

            self._current_state = post
            feat = self.dynamics_model.get_feat(post)
            dist = self.policy_network(feat)

            # Safe mode/sample access
            action = dist.sample() if sample else (dist.mode() if callable(getattr(dist, "mode", None)) else dist.mode)

            # Ensure action is 3D (B, 1, D) for the next observation step
            if action.dim() == 2:
                action = action.unsqueeze(1)
            self._prev_action = action

            # State for RL wrapper must include prev_action
            post["prev_action"] = action

        return action, post

    def report(self, data):
        """Generate comprehensive visualizations (Separated Recon, Imagination, and Latent Analysis)."""
        self.eval()
        from torch_wm.utils.preprocessing import preprocess_obs_for_agent
        import cv2

        def apply_pca(x, n_components=3):
            # x: (N, D)
            if x.shape[0] < n_components:
                return torch.zeros(x.shape[0], n_components, device=x.device)
            x_mean = x - x.mean(0)
            try:
                # Use sklearn for more robust PCA if available, else torch
                from sklearn.decomposition import PCA as skPCA
                pca = skPCA(n_components=n_components)
                projected = torch.from_numpy(pca.fit_transform(x_mean.cpu().numpy())).to(x.device)
                # Normalize to [0, 1] for RGB viewing
                mins = projected.min(0, keepdim=True)[0]
                maxs = projected.max(0, keepdim=True)[0]
                return (projected - mins) / (maxs - mins + 1e-8)
            except ImportError:
                U, S, V = torch.pca_lowrank(x_mean, q=n_components)
                projected = torch.matmul(x_mean, V[:, :n_components])
                mins = projected.min(0, keepdim=True)[0]
                maxs = projected.max(0, keepdim=True)[0]
                return (projected - mins) / (maxs - mins + 1e-8)

        with torch.no_grad():
            # 1. Preprocess inputs
            s = preprocess_obs_for_agent(data, self.device)
            obs = {k: v for k, v in s.items()
                   if k not in ("action", "reward", "is_first", "is_last", "is_terminal")}
            a = s["action"]
            f = s["is_first"].float()
            r = s["reward"]
            if a.dim() == 2: a = a.unsqueeze(-1)
            if f.dim() == 2: f = f.unsqueeze(-1)

            # 2. Forward pass (Posterior)
            encoder_out = self.encoder_network(obs)
            observe_kwargs = {"return_att_w": True} if self.config.get("dynamics_type", "tssm") == "tssm" else {}
            posts, _ = self.dynamics_model.observe(
                encoder_out, a, f, **observe_kwargs
            )

            # 3. Imagination (Prior)
            def slice_first(x):
                if isinstance(x, (list, tuple)): return type(x)(slice_first(i) for i in x)
                if isinstance(x, torch.Tensor): return x[:, 0:1]
                return x
            init_state = {k: slice_first(v) for k, v in posts.items()}
            prior_states = self.dynamics_model.imagine(self.policy_network, init_state, img_steps=self.config.H)

            # 4. Decode — TWISTER parity: decoder gets ONLY stoch, not feats
            post_feats = self.dynamics_model.get_feat(posts)
            prior_feats = self.dynamics_model.get_feat(prior_states)
            post_stoch = posts["stoch"].flatten(-2, -1) if posts["stoch"].dim() > 3 else posts["stoch"]
            prior_stoch = prior_states["stoch"].flatten(-2, -1) if prior_states["stoch"].dim() > 3 else prior_states["stoch"]
            
            if self.config.get("dynamics_type", "tssm") == "rssm":
                post_dec_in = torch.cat([posts["deter"], post_stoch], dim=-1)
                prior_dec_in = torch.cat([prior_states["deter"], prior_stoch], dim=-1)
            else:
                post_dec_in = post_stoch
                prior_dec_in = prior_stoch
                
            post_recs = self.decoder_network(post_dec_in)
            prior_recs = self.decoder_network(prior_dec_in)

            report = {}
            H = self.config.H
            # 5. Process Videos & Latent PCA
            for key in post_recs:
                if key in obs:
                    # ── Type 1: Reconstruction from Replay Buffer (Posterior) ──
                    # Input: real observations → encoder → TSSM.observe → decoder
                    # Shows how well the WM reconstructs SEEN frames
                    gt = (obs[key] + 0.5).clamp(0, 1)
                    rec_dist = post_recs[key]
                    rec = (rec_dist.mode() if callable(rec_dist.mode) else rec_dist.mode)
                    rec = (rec + 0.5).clamp(0, 1)

                    recon_panel = torch.cat([gt, rec], dim=-1)  # GT(left) | Recon(right)
                    report[f"Visual_Consistency/{key}"] = recon_panel[0].permute(0, 2, 3, 1).cpu().numpy()

                    # ── Type 2: Imagination from World Model (Prior) ──
                    # Input: posterior state(t=0) → policy_network → TSSM.imagine(H steps) → decoder
                    # This is the WM's "dream" — NO ground truth exists for imagined futures.
                    # Only the first frame (t=0) comes from real data; frames 1..H are hallucinated.
                    imag_dist = prior_recs[key]
                    imag = (imag_dist.mode() if callable(imag_dist.mode) else imag_dist.mode)
                    imag = (imag + 0.5).clamp(0, 1)
                    report[f"Visual_Imagination/{key}"] = imag[0].permute(0, 2, 3, 1).cpu().numpy()

                    # --- Latent Analysis (PCA) ---
                    if key == "camera" and key in self.encoder_network.encoders:
                        enc = self.encoder_network.encoders[key]
                        spatial_raw = enc.cnn(obs[key].flatten(0, 1))
                        if spatial_raw.dim() == 4:
                            B_L, C_f, H_f, W_f = spatial_raw.shape
                            spatial_flat = spatial_raw.permute(0, 2, 3, 1).reshape(-1, C_f)
                            pca_spatial = apply_pca(spatial_flat, 3)
                            pca_img = pca_spatial.reshape(B_L, H_f, W_f, 3).cpu().numpy()

                            H_orig, W_orig = obs[key].shape[-2:]
                            pca_viz = []
                            for t in range(pca_img.shape[0]):
                                upscaled = cv2.resize(pca_img[t], (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)
                                pca_viz.append(upscaled)
                            pca_viz = np.stack(pca_viz).reshape(obs[key].shape[0], obs[key].shape[1], H_orig, W_orig, 3)
                            report[f"Latent_Analysis/Spatial_PCA_{key}"] = pca_viz[0]

            # ── Imagination Head Predictions (scalars) ──
            # Log predicted reward/value/discount from imagined trajectory
            imag_reward_dist = self.reward_network(prior_feats)
            imag_reward = imag_reward_dist.mode() if callable(imag_reward_dist.mode) else imag_reward_dist.mode
            report["Imagination/Reward_Predicted_Mean"] = imag_reward.mean().item()
            report["Imagination/Reward_Predicted_Std"] = imag_reward.std().item()

            imag_value_dist = self.value_network(prior_feats)
            imag_value = imag_value_dist.mode() if callable(imag_value_dist.mode) else imag_value_dist.mode
            report["Imagination/Value_Predicted_Mean"] = imag_value.mean().item()

            imag_disc_dist = self.continue_network(prior_feats)
            imag_disc = imag_disc_dist.mode if not callable(imag_disc_dist.mode) else imag_disc_dist.mode()
            report["Imagination/Continue_Predicted_Mean"] = imag_disc.float().mean().item()

            # Compare predicted reward vs true reward (from replay buffer)
            true_reward_mean = r.mean().item()
            report["Imagination/True_Reward_Mean"] = true_reward_mean
            report["Imagination/Reward_Gap"] = abs(imag_reward.mean().item() - true_reward_mean)

            # Sequence Latent Analysis (Global)
            B, L, D = post_feats.shape
            seq_flat = post_feats.reshape(-1, D)
            pca_seq = apply_pca(seq_flat, 3)
            pca_seq_img = pca_seq.reshape(B, L, 1, 3).cpu().numpy()
            pca_seq_viz = []
            for t in range(L):
                block = np.zeros((32, 256, 3), dtype=np.float32)
                block[:, :] = pca_seq_img[0, t]
                pca_seq_viz.append(block)
            report[f"Latent_Analysis/Sequence_PCA"] = np.stack(pca_seq_viz)

            # --- NEW: Latent Trajectory Plot ---
            # Project to 2D for scatter plot
            try:
                pca_2d = apply_pca(seq_flat, 2)
                pca_2d_np = pca_2d.reshape(B, L, 2).cpu().numpy()
                rewards_np = r.reshape(B, L).cpu().numpy()

                fig, ax = plt.subplots(figsize=(6, 6))
                for b in range(min(B, 4)):
                    # Plot trajectory for each sequence in batch
                    traj = pca_2d_np[b]
                    sc = ax.scatter(traj[:, 0], traj[:, 1], c=rewards_np[b], cmap='viridis', s=20, alpha=0.6)
                    ax.plot(traj[:, 0], traj[:, 1], alpha=0.3)

                ax.set_title("Latent Trajectory (PCA 2D, colored by Reward)")
                ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
                plt.colorbar(sc, ax=ax, label="Reward")

                # Convert plot to image
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                plot_img = plt.imread(buf)
                report["Latent_Analysis/Trajectory_Scatter"] = plot_img[:, :, :3]
                plt.close(fig)
            except Exception as e:
                log.warning(f"Failed to generate latent scatter plot: {e}")

            # 6. Attention Analysis
            if "att_w" in posts and len(posts["att_w"]) > 0:
                last_attn = posts["att_w"][-1]
                if last_attn.dim() == 4:
                    attn_map = last_attn[0].mean(0).cpu().numpy()
                    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
                    attn_map = (attn_map * 255).astype(np.uint8)
                    attn_map_bgr = cv2.applyColorMap(attn_map, cv2.COLORMAP_VIRIDIS)
                    attn_map_rgb = cv2.cvtColor(attn_map_bgr, cv2.COLOR_BGR2RGB)
                    attn_map_viz = cv2.resize(attn_map_rgb, (256, 256), interpolation=cv2.INTER_NEAREST)
                    report["Latent_Analysis/Attention_Map"] = attn_map_viz

            # 7. Actions & Rewards
            if a.shape[-1] > 1:
                act_indices = a.argmax(dim=-1)
                action_cfg = self.config.get("action", self.config.get("env_params", {}).get("action", {}))
                steer_vals_list = action_cfg.get("discrete_steer", [-0.6, 0.0, 0.6])
                acc_vals_list = action_cfg.get("discrete_acc", [-3.0, 0.0, 3.0])
                n_steer = len(steer_vals_list)
                acc_idx = (act_indices // n_steer).float().clamp(0, len(acc_vals_list) - 1)
                steer_idx = (act_indices % n_steer).float().clamp(0, len(steer_vals_list) - 1)
                acc_vals = torch.tensor(acc_vals_list, device=self.device)
                steer_vals = torch.tensor(steer_vals_list, device=self.device)
                real_acc = acc_vals[acc_idx.long()]
                real_steer = steer_vals[steer_idx.long()]
                report["Stats/Action_Acceleration_Mean"] = real_acc.mean().item()
                report["Stats/Action_Steering_Mean"] = real_steer.mean().item()
                report["Stats/Action_Steering_Histogram"] = real_steer.cpu().numpy()
                report["Stats/Action_Index_Histogram"] = act_indices.float().cpu().numpy()

                # Check for "Standing Still" - fraction of neutral actions
                # Neutral index is typically where acc=0 and steer=0.
                # Find the index for (0.0, 0.0)
                try:
                    zero_acc_idx = acc_vals_list.index(0.0)
                    zero_steer_idx = steer_vals_list.index(0.0)
                    neutral_idx = zero_acc_idx * n_steer + zero_steer_idx
                    is_neutral = (act_indices == neutral_idx).float()
                    report["Stats/Action_Neutral_Fraction"] = is_neutral.mean().item()
                except ValueError:
                    pass

            report["Stats/Reward_Sum"] = s["reward"].sum().item()
            report["Stats/Reward_Mean"] = s["reward"].mean().item()

        self.train()
        return report

    # --- Extracted Sub-Models (SOLID: Single Responsibility) ---
    # Each sub-model is in its own file under torch_wm/models/
    from torch_wm.models.world_model import WorldModel
    from torch_wm.models.actor_model import ActorModel
    from torch_wm.models.critic_model import CriticModel

    # --- Utilities ---
    def update_perc(self, returns):
        low = torch.quantile(returns.detach(), 0.05); high = torch.quantile(returns.detach(), 0.95)
        self.perc_low = 0.99 * self.perc_low + 0.01 * low; self.perc_high = 0.99 * self.perc_high + 0.01 * high
        return self.perc_low.detach(), torch.clip(self.perc_high - self.perc_low, min=1.0).detach()

    def compute_td_lambda(self, rewards, values, discounts):
        interm = rewards + discounts * (1 - self.config.lambda_td) * values
        vals = [values[:, -1]]
        for t in reversed(range(interm.shape[1])): vals.append(interm[:, t] + discounts[:, t] * self.config.lambda_td * vals[-1])
        return torch.stack(list(reversed(vals))[:-1], dim=1)

    def preprocess_inputs(self, x, time_stacked):
        from torch_wm.utils.preprocessing import normalize_image
        if isinstance(x, torch.Tensor):
            return normalize_image(x)
        return x

    def compile(self):
        self._last_outputs = {}

        # 1. Perform dummy forward pass to initialize LazyLinear layers
        self._perform_dry_run()

        model_params = list(itertools.chain(
            self.encoder_network.parameters(),
            self.dynamics_model.parameters(),
            self.reward_network.parameters(),
            self.decoder_network.parameters(),
            self.continue_network.parameters(),
            self.contrastive_network.parameters(),
            self.world_model.loss_manager.parameters()
        ))

        # ── Optimizer & Scheduler (DreamerV3 Style) ──
        def get_opt_kwargs(key):
            cfg = self.config.get(key, {})
            return {
                "lr": cfg.get("lr", 1e-4),
                "eps": float(cfg.get("eps", 1e-8)),
                "wd": float(cfg.get("wd", 0.0)),
                "clip": float(cfg.get("clip", 1000.0)),
                "warmup": int(cfg.get("warmup", 0)),
                "decay_steps": int(float(cfg.get("decay_steps", 1e6))),
                "min_lr": float(cfg.get("min_lr", 1e-5)),
            }

        m_opt = get_opt_kwargs("model_opt")
        a_opt = get_opt_kwargs("actor_opt")
        c_opt = get_opt_kwargs("critic_opt")

        wm_lr = LinearWarmupCosineScheduler(m_opt["lr"], m_opt["warmup"], m_opt["decay_steps"], m_opt["min_lr"] / m_opt["lr"])
        actor_lr = LinearWarmupCosineScheduler(a_opt["lr"], a_opt["warmup"], a_opt["decay_steps"], a_opt["min_lr"] / a_opt["lr"])
        critic_lr = LinearWarmupCosineScheduler(c_opt["lr"], c_opt["warmup"], c_opt["decay_steps"], c_opt["min_lr"] / c_opt["lr"])

        self.world_model.compile(optimizer=Adam(model_params, lr=wm_lr, eps=m_opt["eps"], weight_decay=m_opt["wd"], grad_max_norm=m_opt["clip"]), losses=None)
        self.actor_model.compile(optimizer=Adam(list(self.policy_network.parameters()), lr=actor_lr, eps=a_opt["eps"], weight_decay=a_opt["wd"], grad_max_norm=a_opt["clip"]), losses=None)
        self.critic_model.compile(optimizer=Adam(list(self.value_network.parameters()), lr=critic_lr, eps=c_opt["eps"], weight_decay=c_opt["wd"], grad_max_norm=c_opt["clip"]), losses=None)
        log.info(f"[OPT] WM lr={m_opt['lr']}, Actor lr={a_opt['lr']}, Critic lr={c_opt['lr']} | Actor eps={a_opt['eps']}")

        self.model_step = self.world_model.optimizer.param_groups[0]["lr_scheduler"].model_step
        self.optimizer = {"wm": self.world_model.optimizer, "actor": self.actor_model.optimizer, "critic": self.critic_model.optimizer}

        self.built = True
        self.compiled = True

        # ══════════════════════════════════════════════════════════════
        # DETAILED MODULE LOADING LOG — Rich Tables
        # ══════════════════════════════════════════════════════════════
        from rich.table import Table
        from rich.panel import Panel
        from rich import box
        from torch_wm.utils import get_console
        console = get_console()

        def _count_params(module, trainable_only=False):
            if trainable_only:
                return sum(p.numel() for p in module.parameters() if p.requires_grad)
            return sum(p.numel() for p in module.parameters())

        # ── 1. Networks Table ──
        t = Table(title="[bold magenta]🧠 Module Loading Summary[/bold magenta]", box=box.DOUBLE_EDGE)
        t.add_column("Module", style="cyan", width=26)
        t.add_column("Type", style="white", width=24)
        t.add_column("Params", style="green", justify="right", width=12)
        t.add_column("Trainable", style="yellow", justify="right", width=12)
        t.add_column("Status", width=8)

        nets = [
            ("encoder_network", self.encoder_network),
            ("decoder_network", self.decoder_network),
            ("dynamics_model", self.dynamics_model),
            ("policy_network", self.policy_network),
            ("value_network", self.value_network),
            ("reward_network", self.reward_network),
            ("continue_network", self.continue_network),
            ("contrastive_network", self.contrastive_network),
            ("v_target", self.v_target),
        ]
        total_trainable = 0
        total_all = 0
        for name, net in nets:
            cls_name = type(net).__name__
            n_all = _count_params(net)
            n_train = _count_params(net, trainable_only=True)
            status = "❄️ Frozen" if n_train == 0 else "✅ Active"
            t.add_row(name, cls_name, f"{n_all:,}", f"{n_train:,}", status)
            total_all += n_all
            total_trainable += n_train
        t.add_section()
        t.add_row("[bold]TOTAL[/bold]", "", f"[bold]{total_all:,}[/bold]", f"[bold]{total_trainable:,}[/bold]", "")
        console.print(t)

        # ── 2. Sub-Models ──
        t2 = Table(title="[bold blue]🔄 Training Phases[/bold blue]", box=box.ROUNDED)
        t2.add_column("Phase", style="cyan", width=16)
        t2.add_column("Model Class", style="white", width=28)
        t2.add_column("Optimized Networks", style="green", width=36)
        t2.add_row("1. World Model", type(self.world_model).__name__, "encoder, decoder, dynamics, reward, continue, cpc")
        t2.add_row("2. Actor", type(self.actor_model).__name__, "policy_network")
        t2.add_row("3. Critic", type(self.critic_model).__name__, "value_network")
        console.print(t2)

        # ── 3. Policy & Action Space ──
        policy_type = self.config.get("policy_type", "default_policy")
        actor_model_type = self.config.get("actor_model_type", "default_actor")
        disc_acc = self.config.get("discrete_acc", [])
        disc_steer = self.config.get("discrete_steer", [])

        t3 = Table(title="[bold green]🎮 Policy & Action Space[/bold green]", box=box.DOUBLE_EDGE)
        t3.add_column("Parameter", style="cyan", width=22)
        t3.add_column("Value", style="white", width=50)
        t3.add_row("policy_type", f"[bold]{policy_type}[/bold]")
        t3.add_row("actor_model_type", f"[bold]{actor_model_type}[/bold]")
        t3.add_row("num_actions", f"{self.num_actions} ({len(disc_acc)} acc × {len(disc_steer)} steer)")
        t3.add_row("discrete_acc", str(disc_acc))
        t3.add_row("discrete_steer", str(disc_steer))
        t3.add_row("uniform_mix", str(self.config.get("uniform_mix", 0.01)))
        console.print(t3)

        # ── 4. RL Hyperparameters ──
        gamma = self.config.get("gamma", 0.997)
        H = self.config.get("H", 15)
        lambda_td = self.config.get("lambda_td", 0.95)
        actent = self.config.get("actent", 0.0003)
        tvr = self.config.get("target_value_reg", False)
        actor_grad = self.config.get("run", {}).get("actor_grad_disc", "reinforce")

        t4 = Table(title="[bold yellow]⚙️  RL Hyperparameters[/bold yellow]", box=box.DOUBLE_EDGE)
        t4.add_column("Parameter", style="cyan", width=22)
        t4.add_column("Value", style="white", width=18)
        t4.add_column("Description", style="dim", width=30)
        t4.add_row("gamma", str(gamma), "Discount factor")
        t4.add_row("lambda_td", str(lambda_td), "TD-λ trace decay")
        t4.add_row("H", str(H), "Imagination horizon")
        t4.add_row("actent", str(actent), "Entropy bonus scale")
        t4.add_row("target_value_reg", str(tvr), "Critic slow-reg enabled")
        t4.add_row("actor_grad", actor_grad, "Policy gradient method")
        t4.add_row("critic_ema_decay", "0.02", "v_target EMA rate")
        console.print(t4)

        # ── 5. Optimizer Config ──
        t5 = Table(title="[bold red]📉 Optimizer Configuration[/bold red]", box=box.DOUBLE_EDGE)
        t5.add_column("Component", style="cyan", width=15)
        t5.add_column("LR", style="green", justify="right", width=12)
        t5.add_column("EPS", style="yellow", justify="right", width=12)
        t5.add_column("Grad Clip", style="white", justify="right", width=10)
        t5.add_column("Weight Decay", style="dim", justify="right", width=12)
        t5.add_row("World Model", f"{m_opt['lr']:.6f}", f"{m_opt['eps']:.1e}", f"{m_opt['clip']:.0f}", f"{m_opt['wd']:.4f}")
        t5.add_row("Actor", f"{a_opt['lr']:.6f}", f"{a_opt['eps']:.1e}", f"{a_opt['clip']:.0f}", f"{a_opt['wd']:.4f}")
        t5.add_row("Critic", f"{c_opt['lr']:.6f}", f"{c_opt['eps']:.1e}", f"{c_opt['clip']:.0f}", f"{c_opt['wd']:.4f}")
        console.print(t5)

        # ── 6. Dynamics Model ──
        dynamics_type = self.config.get("dynamics_type", "tssm")
        feat_size = self.config.stoch_size * self.config.get("discrete", 32) + self.config.get("hidden_size", 256)

        t6 = Table(title="[bold cyan]🌀 Dynamics Model[/bold cyan]", box=box.DOUBLE_EDGE)
        t6.add_column("Parameter", style="cyan", width=22)
        t6.add_column("Value", style="white", width=50)
        t6.add_row("type", dynamics_type)
        t6.add_row("hidden_size (deter)", str(self.config.get("hidden_size", 256)))
        t6.add_row("stoch_size", str(self.config.get("stoch_size", 32)))
        t6.add_row("discrete", str(self.config.get("discrete", 32)))
        t6.add_row("att_context_left", str(self.config.get("att_context_left", 32)))
        t6.add_row("feat_size", f"[bold]{feat_size}[/bold]  (stoch×discrete + hidden = {self.config.get('stoch_size',32)}×{self.config.get('discrete',32)} + {self.config.get('hidden_size',256)})")
        console.print(t6)

        # ── 7. Loss Modules ──
        modules_cfg = self.config.get("modules", {}).get("losses", {})
        t7 = Table(title="[bold red]📊 Loss Functions[/bold red]", box=box.DOUBLE_EDGE)
        t7.add_column("Loss", style="cyan", width=22)
        t7.add_column("Status", width=10)
        t7.add_column("Weight", style="yellow", justify="right", width=10)
        for name, cfg in modules_cfg.items():
            enabled = cfg.get("enabled", False)
            weight = cfg.get("weight", 0.0)
            status = "[green]✅ Active[/green]" if enabled else "[dim]❌ Off[/dim]"
            t7.add_row(name, status, str(weight))
        console.print(t7)

    def train_step(self, inputs, targets, precision, grad_scaler, accumulated_steps, acc_step, eval_training):
        inputs = self.preprocess_inputs(inputs, True)
        # 1. World Model
        self.set_require_grad([self.policy_network, self.value_network], False)
        self.set_require_grad([self.encoder_network, self.decoder_network, self.dynamics_model, self.reward_network, self.continue_network], True)
        wm_loss, _, _ = self.world_model.train_step(inputs, targets, precision, grad_scaler, accumulated_steps, acc_step, eval_training)

        # 2. Actor (TWISTER lines 724-729)
        self.dynamics_model.eval()
        self.set_require_grad(self.policy_network, True)
        self.set_require_grad([self.value_network, self.encoder_network, self.decoder_network,
                               self.dynamics_model, self.reward_network, self.continue_network], False)
        act_loss, _, _ = self.actor_model.train_step(inputs, targets, precision, grad_scaler, accumulated_steps, acc_step, eval_training)
        # 3. Critic (TWISTER lines 738-740)
        self.set_require_grad(self.value_network, True)
        self.set_require_grad([self.policy_network, self.encoder_network, self.decoder_network,
                               self.dynamics_model, self.reward_network, self.continue_network], False)
        crit_loss, _, _ = self.critic_model.train_step(inputs, targets, precision, grad_scaler, accumulated_steps, acc_step, eval_training)

        self.dynamics_model.train(); self.update_target_networks()

        # Return raw loss dicts to allow UnifiedAgent to handle prefixing/grouping
        return {
            "wm": wm_loss,
            "actor": act_loss,
            "critic": crit_loss
        }, self.world_model.loss_manager.get_metrics() if hasattr(self.world_model, "loss_manager") else {}, _

    def eval_step(self, inputs, targets, verbose):
        inputs = self.preprocess_inputs(inputs, True)

        # 1. World Model
        wm_loss, _, _, _ = self.world_model.eval_step(inputs, targets, verbose)
        # 2. Actor
        act_loss, _, _, _ = self.actor_model.eval_step(inputs, targets, verbose)
        # 3. Critic
        crit_loss, _, _, _ = self.critic_model.eval_step(inputs, targets, verbose)

        metrics = self.world_model.loss_manager.get_metrics() if hasattr(self.world_model, "loss_manager") else {}

        # Rename keys to avoid overwriting "loss"
        final_losses = {}
        for k, v in wm_loss.items(): final_losses[f"wm_{k}"] = v
        for k, v in act_loss.items(): final_losses[f"actor_{k}"] = v
        for k, v in crit_loss.items(): final_losses[f"critic_{k}"] = v

        return final_losses, metrics, None, None

    def update_target_networks(self):
        for p_t, p_n in zip(self.v_target.parameters(), self.value_network.parameters()):
            p_t.mul_(0.98).add_(0.02 * p_n.detach())

    def log_figure(self, step, inputs, targets, writer, tag):
        # Implementation for projected trajectory rendering on WandB
        if writer is None or not hasattr(self.env.envs[0], "camera_K"): return
        self.eval(); inputs = self.preprocess_inputs(inputs, True)
        with torch.no_grad():
            latent = self.encoder_network(inputs[0][:4])
            posts, _ = self.dynamics_model.observe(latent, inputs[1][:4], inputs[4][:4])
            if self.config.get("dynamics_type", "tssm") == "rssm":
                dec_in = torch.cat([posts["deter"], posts["stoch"].flatten(-2, -1)], dim=-1)
            else:
                dec_in = posts["stoch"].flatten(-2, -1)
            rec = self.decoder_network(dec_in)["camera"].mode

        if HAS_TORCHVISION:
            grid = torchvision.utils.make_grid((rec + 0.5).clip(0, 1), nrow=8)
            writer.add_image(tag, grid, step)
        else:
            print("[Warning] Skip log_figure because torchvision is missing.")
        self.train()

    def _perform_dry_run(self):
        """Perform a small dummy forward pass to initialize LazyLinear layers."""
        from torch_wm.structs import AttrDict
        batch_size = 1
        batch_length = 2

        dummy_batch = AttrDict()
        # 1. Image Modalities from Encoders
        encoders = getattr(self.encoder_network, "encoders", {})
        if encoders:
            for name, encoder in encoders.items():
                h, w = encoder.image_size
                c = getattr(encoder, "dim_input_cnn", 3)
                dummy_batch[name] = torch.zeros(batch_size, batch_length, c, h, w, device=self.device)
        elif hasattr(self.encoder_network, "image_size"):
            # Single encoder (e.g. V-JEPA)
            h, w = self.encoder_network.image_size
            dummy_batch["camera"] = torch.zeros(batch_size, batch_length, 3, h, w, device=self.device)
        else:
            # Absolute fallback
            dummy_batch["camera"] = torch.zeros(batch_size, batch_length, 3, 64, 64, device=self.device)

        # 2. Standard RL keys
        dummy_batch["is_first"] = torch.zeros(batch_size, batch_length, device=self.device)
        dummy_batch["action"] = torch.zeros(batch_size, batch_length, self.num_actions, device=self.device)
        dummy_batch["reward"] = torch.zeros(batch_size, batch_length, device=self.device)
        dummy_batch["discount"] = torch.ones(batch_size, batch_length, device=self.device)

        # Preprocess images (converts to [-0.5, 0.5] if needed)
        for k in dummy_batch:
            if hasattr(self.encoder_network, "encoders") and k in self.encoder_network.encoders:
                dummy_batch[k] = self.preprocess_inputs(dummy_batch[k], True)

        # 3. Trigger forward pass to initialize LazyLinear in LossManager/JEPA
        with torch.no_grad():
            # s: dict of sensors, a: actions, r: rewards, d: dones, f: is_firsts
            if hasattr(self.encoder_network, "encoders"):
                s_dict = {k: v for k, v in dummy_batch.items() if k in self.encoder_network.encoders}
            else:
                s_dict = {"camera": dummy_batch.get("camera")} # Fallback for single image encoder like V-JEPA
            a = dummy_batch.action
            r = dummy_batch.reward
            d = 1.0 - dummy_batch.discount
            f = dummy_batch.is_first

            self.world_model((s_dict, a, r, d, f))
