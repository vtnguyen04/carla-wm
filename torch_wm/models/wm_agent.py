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
from torch import nn
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

        self.decoder_network = networks.MultiDecoderNetwork(
            obs_config=obs_config,
            obs_space=obs_space,
            feat_size=feat_size,
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


        self.policy_network = networks.PolicyNetwork(
            num_actions,
            **get_head_kwargs("actor", {"hidden_size": self.config.hidden_size, "num_mlp_layers": 5}, include_bins=False),
            feat_size=feat_size,
            discrete=self.config.discrete_actions,
            uniform_mix=self.config.uniform_mix,
        )
        self.value_network = networks.ValueNetwork(
            **get_head_kwargs("critic", {"hidden_size": self.config.hidden_size, "num_mlp_layers": 5}),
            feat_size=feat_size
        )
        self.reward_network = networks.RewardNetwork(
            **get_head_kwargs("reward_head", {"hidden_size": self.config.hidden_size, "num_mlp_layers": 5}),
            feat_size=feat_size
        )
        self.continue_network = networks.ContinueNetwork(
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
        self.actor_model = self.ActorModel(outer=self)
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
            # obs is (B, ...) from environment, usually needs unsqueeze(1)
            latent_step = {k: v.unsqueeze(1) if v.dim() == latent['stoch'].dim() else v for k, v in latent.items()}
            
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
            if sample:
                action = dist.sample()
            else:
                action = dist.mode() if callable(dist.mode) else dist.mode
                
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
            observe_kwargs = {"return_att_w": True} if self.config.dynamics_type == "tssm" else {}
            posts, _ = self.dynamics_model.observe(
                {"stoch": encoder_out["stoch"], "logits": encoder_out["logits"]}, a, f, **observe_kwargs
            )
            
            # 3. Imagination (Prior)
            def slice_first(x):
                if isinstance(x, (list, tuple)): return type(x)(slice_first(i) for i in x)
                if isinstance(x, torch.Tensor): return x[:, 0:1]
                return x
            init_state = {k: slice_first(v) for k, v in posts.items()}
            prior_states = self.dynamics_model.imagine(self.policy_network, init_state, img_steps=self.config.H)
            
            # 4. Decode
            post_feats = self.dynamics_model.get_feat(posts)
            prior_feats = self.dynamics_model.get_feat(prior_states)
            post_recs = self.decoder_network(post_feats)
            prior_recs = self.decoder_network(prior_feats)
            
            report = {}
            # 5. Process Videos & Latent PCA
            for key in post_recs:
                if key in obs:
                    # GT | Reconstruction
                    gt = (obs[key] + 0.5).clamp(0, 1)
                    rec_dist = post_recs[key]
                    rec = (rec_dist.mode() if callable(rec_dist.mode) else rec_dist.mode)
                    rec = (rec + 0.5).clamp(0, 1)
                    
                    recon_panel = torch.cat([gt, rec], dim=-1)
                    report[f"Visual_Consistency/{key}"] = recon_panel[0].permute(0, 2, 3, 1).cpu().numpy()
                    
                    # Imagination panel
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
                print(f"[Warning] Failed to generate latent scatter plot: {e}")

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
                steer_vals_list = self.config.get("discrete_steer", [-0.6, 0.0, 0.6])
                acc_vals_list = self.config.get("discrete_acc", [-3.0, 0.0, 3.0])
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
        
        self.model_step = self.world_model.optimizer.param_groups[0]["lr_scheduler"].model_step
        self.optimizer = {"wm": self.world_model.optimizer, "actor": self.actor_model.optimizer, "critic": self.critic_model.optimizer}

        self.built = True
        self.compiled = True

    def train_step(self, inputs, targets, precision, grad_scaler, accumulated_steps, acc_step, eval_training):
        inputs = self.preprocess_inputs(inputs, True)
        # 1. World Model
        self.set_require_grad([self.policy_network, self.value_network], False)
        self.set_require_grad([self.encoder_network, self.decoder_network, self.dynamics_model, self.reward_network, self.continue_network], True)
        wm_loss, _, _ = self.world_model.train_step(inputs, targets, precision, grad_scaler, accumulated_steps, acc_step, eval_training)
        # 2. Actor
        self.dynamics_model.eval(); self.set_require_grad(self.policy_network, True)
        act_loss, _, _ = self.actor_model.train_step(inputs, targets, precision, grad_scaler, accumulated_steps, acc_step, eval_training)
        # 3. Critic
        self.set_require_grad(self.value_network, True); self.set_require_grad(self.policy_network, False)
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
            rec = self.decoder_network(posts["stoch"].flatten(-2, -1))["camera"].mode
        
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
        for name, encoder in self.encoder_network.encoders.items():
            h, w = encoder.image_size
            c = encoder.dim_input_cnn
            dummy_batch[name] = torch.zeros(batch_size, batch_length, c, h, w, device=self.device)
        
        # 2. Standard RL keys
        dummy_batch["is_first"] = torch.zeros(batch_size, batch_length, device=self.device)
        dummy_batch["action"] = torch.zeros(batch_size, batch_length, self.num_actions, device=self.device)
        dummy_batch["reward"] = torch.zeros(batch_size, batch_length, device=self.device)
        dummy_batch["discount"] = torch.ones(batch_size, batch_length, device=self.device)
        
        # Preprocess images (converts to [-0.5, 0.5] if needed)
        for k in dummy_batch:
            if k in self.encoder_network.encoders:
                dummy_batch[k] = self.preprocess_inputs(dummy_batch[k], True)

        # 3. Trigger forward pass to initialize LazyLinear in LossManager/JEPA
        with torch.no_grad():
            # Construct tuple for WorldModel.forward(self, inputs)
            # s: dict of sensors, a: actions, r: rewards, d: dones, f: is_firsts
            s_dict = {k: v for k, v in dummy_batch.items() if k in self.encoder_network.encoders}
            a = dummy_batch.action
            r = dummy_batch.reward
            d = 1.0 - dummy_batch.discount
            f = dummy_batch.is_first
            
            self.world_model((s_dict, a, r, d, f))
