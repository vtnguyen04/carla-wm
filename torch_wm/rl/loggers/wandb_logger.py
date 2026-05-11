"""
Weights & Biases Logger — Full experiment tracking with video reconstruction.
Single Responsibility: All WandB API calls live here.

Features:
  - Reconstruction video: GT vs Reconstructed side-by-side for full sequences
  - Imagination rollout video: Multi-step predicted futures
  - Per-branch image grids with MSE error heatmaps
  - Reward distribution histograms (true vs predicted)
  - Latent space statistics (entropy, sparsity, KL divergence)
  - Gradient norm tracking per component
  - Model architecture summary table
  - Action distribution analysis
"""
import datetime
import tempfile
import pathlib
import numpy as np
import torch


class WandBLogger:
    """Professional WandB integration with video reconstruction and deep analytics."""

    def __init__(self, config: dict, log_dir: str, project: str = "twister-carla"):
        self.enabled = False
        self._run = None
        self._wandb = None
        try:
            import wandb
            self._wandb = wandb
            self._run = wandb.init(
                project=project,
                name=f"offline-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=config,
                dir=log_dir,
                reinit=True,
                settings=wandb.Settings(silent=True),
            )
            self.enabled = True
        except Exception as e:
            print(f"[WandB] Init failed: {e}")

    # ═══════════════════════════════════════════
    #  SCALARS
    # ═══════════════════════════════════════════

    def log_step(self, metrics: dict, global_step: int):
        """Log per-step metrics."""
        if not self.enabled:
            return
        log_dict = {}
        for k, v in metrics.items():
            val = v.item() if isinstance(v, torch.Tensor) else float(v)
            # Group by prefix for cleaner WandB panels
            if "loss" in k:
                log_dict[f"losses/{k}"] = val
            elif "cpc" in k:
                log_dict[f"cpc/{k}"] = val
            else:
                log_dict[f"step/{k}"] = val
        self._wandb.log(log_dict, step=global_step)

    def log_epoch(self, metrics: dict, epoch: int, global_step: int,
                  speed: float = 0.0, gpu_mem: float = 0.0):
        """Log epoch-level averaged metrics with grouped panels."""
        if not self.enabled:
            return
        log_dict = {}
        for k, v in metrics.items():
            val = float(v)
            if "loss" in k:
                log_dict[f"losses/{k}"] = val
            elif "cpc" in k:
                log_dict[f"cpc/{k}"] = val
            elif "wm_" in k:
                log_dict[f"world_model/{k}"] = val
            elif "actor" in k:
                log_dict[f"actor/{k}"] = val
            elif "critic" in k:
                log_dict[f"critic/{k}"] = val
            else:
                log_dict[f"metrics/{k}"] = val

        log_dict["training/epoch"] = epoch
        log_dict["training/speed_it_s"] = speed
        log_dict["system/gpu_memory_gb"] = gpu_mem
        self._wandb.log(log_dict, step=global_step)

    # ═══════════════════════════════════════════
    #  RECONSTRUCTION VIDEO (GT vs Reconstructed)
    # ═══════════════════════════════════════════

    def log_reconstructions(self, model, batch: dict, global_step: int,
                            num_frames: int = 16):
        """
        Log GT vs Reconstructed side-by-side video + image grid + error maps.
        Uses model._last_outputs['states_rec_dist'] from the training forward pass
        (already correctly computed with full feat = stoch + hidden).
        """
        if not self.enabled:
            return

        log_dict = {}

        try:
            with torch.no_grad():
                # Get reconstructions from last training forward pass
                last = getattr(model, "_last_outputs", {})
                rec_dists = last.get("states_rec_dist")
                if rec_dists is None:
                    print("[WandB] No reconstruction available (states_rec_dist missing)")
                    return

                # If MultiDecoderNetwork returns a dict of distributions
                if not isinstance(rec_dists, dict):
                    # Single decoder — wrap in dict
                    rec_dists = {"camera": rec_dists}

                # Process each branch
                for branch_name, rec_dist in rec_dists.items():
                    raw = batch.get(branch_name)
                    if raw is None or not isinstance(raw, torch.Tensor):
                        continue
                    if raw.dim() < 4:
                        continue

                    # MSEDist.mode() is a method, not a property
                    if hasattr(rec_dist, 'mode') and callable(rec_dist.mode):
                        rec = rec_dist.mode()
                    elif hasattr(rec_dist, 'mean'):
                        rec = rec_dist.mean if not callable(rec_dist.mean) else rec_dist.mean()
                    else:
                        continue

                    B, L = raw.shape[0], raw.shape[1]
                    N = min(num_frames, L)

                    # Normalize: preprocessed batch is [-0.5, 0.5], convert to [0, 255] uint8
                    raw_seq = ((raw[0, :N] + 0.5).clamp(0, 1) * 255).byte()   # (N, C, H, W)
                    rec_seq = ((rec[0, :N] + 0.5).clamp(0, 1) * 255).byte()   # (N, C, H, W)

                    # ── 1. Side-by-side VIDEO ──
                    video_frames = torch.cat([raw_seq, rec_seq], dim=-1)  # (N, C, H, 2W)
                    log_dict[f"video/{branch_name}_gt_vs_rec"] = self._wandb.Video(
                        video_frames.cpu().numpy(), fps=4,
                        caption=f"{branch_name}: Left=GT | Right=Reconstructed",
                    )

                    # ── 2. Image grid: side-by-side per timestep ──
                    grid_images = []
                    for t in range(N):
                        gt_np = raw_seq[t].permute(1, 2, 0).cpu().numpy()
                        rc_np = rec_seq[t].permute(1, 2, 0).cpu().numpy()
                        combined = np.concatenate([gt_np, rc_np], axis=1)
                        grid_images.append(
                            self._wandb.Image(combined, caption=f"t={t} (GT|Rec)")
                        )
                    log_dict[f"frames/{branch_name}_sequence"] = grid_images

                    # ── 3. MSE Error Heatmap ──
                    for idx, label in [(0, "first"), (min(N-1, N), "last")]:
                        raw_f = (raw[0, idx] + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                        rec_f = (rec[0, idx] + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                        mse_map = np.mean((raw_f - rec_f) ** 2, axis=-1)
                        mse_vis = (mse_map / (mse_map.max() + 1e-8) * 255).astype(np.uint8)
                        log_dict[f"error/{branch_name}_{label}_mse"] = self._wandb.Image(
                            mse_vis, caption=f"MSE Error t={idx}"
                        )

                    # ── 4. Per-branch MSE scalar ──
                    raw_float = raw[0, :N].float()
                    rec_float = rec[0, :N].float()
                    log_dict[f"reconstruction/{branch_name}_mse"] = (
                        (raw_float - rec_float) ** 2
                    ).mean().item()

            if log_dict:
                self._wandb.log(log_dict, step=global_step)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[WandB] Reconstruction logging failed: {e}")


    def log_imagination(self, model, global_step: int, num_steps: int = 15):
        """
        Generate multi-step imagination rollout video from the policy.
        Uses detached_posts stored from the last training forward pass.
        """
        if not self.enabled:
            return

        try:
            with torch.no_grad():
                # Need hidden state for imagination
                last = getattr(model, "_last_outputs", {})
                detached = getattr(model, "detached_posts", None)
                
                if detached is None or "stoch" not in detached:
                    return

                # Take first sample from batch (B=1), imagine H steps
                # detached has shape (B*L, 1, D) or similar
                # We want to start imagination from a representative point
                H = num_steps
                
                # Create start state for imagination
                start_state = {k: v[:1] if k != "hidden" else [(kk[:1], vv[:1]) for kk, vv in v] for k, v in detached.items()}
                
                # Mock is_firsts
                is_firsts = torch.zeros(1, 1, 1, device=detached["stoch"].device)
                is_firsts_hidden = torch.zeros(1, 1, 1, device=detached["stoch"].device)

                # Dynamics imagine: (1, H, D)
                img_states = model.dynamics_model.imagine(
                    model.policy_network,
                    start_state,
                    H, is_firsts, is_firsts_hidden,
                )

                # Decode imagined states
                # feat = stoch + hidden
                feat = model.dynamics_model.get_feat(img_states)
                img_rec = model.decoder_network(feat)

                log_dict = {}
                for branch_name, dist in img_rec.items():
                    if hasattr(dist, 'mode') and callable(dist.mode):
                        img_frames = dist.mode()
                    elif hasattr(dist, 'mean'):
                        img_frames = dist.mean if not callable(dist.mean) else dist.mean()
                    else:
                        continue
                        
                    # img_frames shape is (1, H+1, C, H, W)
                    img_frames = ((img_frames[0] + 0.5).clamp(0, 1) * 255).byte()  # (1+H, C, H, W)

                    log_dict[f"video/{branch_name}_imagination"] = self._wandb.Video(
                        img_frames.cpu().numpy(),
                        fps=4,
                        caption=f"{branch_name}: {H}-step imagination rollout",
                    )
                
                if log_dict:
                    self._wandb.log(log_dict, step=global_step)

        except Exception as e:
            # Imagination is best-effort, but log error to console if it fails
            print(f"[WandB] Imagination rollout failed: {e}")

    # ═══════════════════════════════════════════
    #  REWARD ANALYSIS
    # ═══════════════════════════════════════════

    def log_reward_analysis(self, batch: dict, model_outputs: dict, global_step: int):
        """Log true vs predicted reward distributions and temporal curves."""
        if not self.enabled:
            return

        try:
            log_dict = {}
            rewards_true = batch.get("reward")
            rewards_pred = model_outputs.get("model_rewards")

            if rewards_true is not None:
                if isinstance(rewards_true, torch.Tensor):
                    rt = rewards_true.detach().cpu().numpy().flatten()
                else:
                    rt = np.asarray(rewards_true).flatten()
                log_dict["rewards/true_histogram"] = self._wandb.Histogram(rt)
                log_dict["rewards/true_mean"] = float(np.mean(rt))
                log_dict["rewards/true_std"] = float(np.std(rt))
                log_dict["rewards/true_min"] = float(np.min(rt))
                log_dict["rewards/true_max"] = float(np.max(rt))

                # Temporal reward curve (first sequence)
                r_arr = rt.reshape(rewards_true.shape[0] if hasattr(rewards_true, 'shape') else -1, -1) if rt.ndim == 1 else rt
                if r_arr.ndim >= 2:
                    seq = r_arr[0]
                    data = [[t, float(r)] for t, r in enumerate(seq)]
                    table = self._wandb.Table(data=data, columns=["timestep", "reward"])
                    log_dict["rewards/true_temporal"] = self._wandb.plot.line(
                        table, "timestep", "reward",
                        title="True Reward Over Sequence"
                    )

            if rewards_pred is not None:
                if hasattr(rewards_pred, 'mode') and callable(rewards_pred.mode):
                    rp = rewards_pred.mode()
                elif hasattr(rewards_pred, 'mean') and callable(rewards_pred.mean):
                    rp = rewards_pred.mean()
                elif isinstance(rewards_pred, torch.Tensor):
                    rp = rewards_pred
                else:
                    rp = None
                if rp is not None and isinstance(rp, torch.Tensor):
                    rp = rp.detach().cpu().numpy().flatten()
                if rp is not None:
                    log_dict["rewards/pred_histogram"] = self._wandb.Histogram(rp)
                    log_dict["rewards/pred_mean"] = float(np.mean(rp))
                    log_dict["rewards/pred_std"] = float(np.std(rp))

                    # Prediction error
                    if rewards_true is not None:
                        rt_flat = rewards_true.detach().cpu().numpy().flatten()
                        min_len = min(len(rt_flat), len(rp))
                        mae = float(np.mean(np.abs(rt_flat[:min_len] - rp[:min_len])))
                        log_dict["rewards/prediction_mae"] = mae

            if log_dict:
                self._wandb.log(log_dict, step=global_step)

        except Exception as e:
            print(f"[WandB] Reward analysis failed: {e}")

    # ═══════════════════════════════════════════
    #  LATENT SPACE ANALYTICS
    # ═══════════════════════════════════════════

    def log_latent_stats(self, model_outputs: dict, global_step: int):
        """Log posterior/prior entropy, sparsity, and activation statistics."""
        if not self.enabled:
            return

        try:
            log_dict = {}
            posts = model_outputs.get("posts", {})
            priors = model_outputs.get("priors", {})

            if "logits" in posts:
                post_logits = posts["logits"].detach()
                post_probs = torch.softmax(post_logits, dim=-1)
                entropy = -(post_probs * (post_probs + 1e-8).log()).sum(dim=-1).mean()
                log_dict["latent/posterior_entropy"] = entropy.item()

                # Max probability (how peaked the distribution is)
                max_prob = post_probs.max(dim=-1).values.mean()
                log_dict["latent/posterior_max_prob"] = max_prob.item()

            if "logits" in priors:
                prior_logits = priors["logits"].detach()
                prior_probs = torch.softmax(prior_logits, dim=-1)
                prior_entropy = -(prior_probs * (prior_probs + 1e-8).log()).sum(dim=-1).mean()
                log_dict["latent/prior_entropy"] = prior_entropy.item()

                # KL divergence (post || prior) per-sample
                if "logits" in posts:
                    post_p = torch.softmax(posts["logits"].detach(), dim=-1)
                    prior_p = prior_probs
                    kl = (post_p * ((post_p + 1e-8).log() - (prior_p + 1e-8).log())).sum(dim=-1).mean()
                    log_dict["latent/kl_divergence"] = kl.item()

            if "stoch" in posts:
                stoch = posts["stoch"].detach()
                stoch_flat = stoch.flatten()
                log_dict["latent/stoch_mean"] = stoch_flat.mean().item()
                log_dict["latent/stoch_std"] = stoch_flat.std().item()
                log_dict["latent/stoch_sparsity"] = (stoch_flat.abs() < 0.01).float().mean().item()
                log_dict["latent/stoch_histogram"] = self._wandb.Histogram(
                    stoch_flat.cpu().numpy()[:10000]  # Cap to avoid huge uploads
                )

            if log_dict:
                self._wandb.log(log_dict, step=global_step)

        except Exception as e:
            print(f"[WandB] Latent stats failed: {e}")

    # ═══════════════════════════════════════════
    #  GRADIENT ANALYSIS
    # ═══════════════════════════════════════════

    def log_gradient_norms(self, model, global_step: int):
        """Log gradient L2 norms per major component."""
        if not self.enabled:
            return

        try:
            log_dict = {}
            components = {
                "encoder": model.encoder_network,
                "decoder": model.decoder_network,
                "tssm": model.dynamics_model,
                "policy": model.policy_network,
                "value": model.value_network,
                "reward": model.reward_network,
                "loss_manager": model.world_model.loss_manager,
            }
            for name, module in components.items():
                total_norm = 0.0
                count = 0
                for p in module.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                        count += 1
                if count > 0:
                    total_norm = total_norm ** 0.5
                    log_dict[f"gradients/{name}_norm"] = total_norm
                    log_dict[f"gradients/{name}_num_params"] = count

            if log_dict:
                self._wandb.log(log_dict, step=global_step)

        except Exception as e:
            print(f"[WandB] Gradient norms failed: {e}")

    # ═══════════════════════════════════════════
    #  ACTION ANALYSIS
    # ═══════════════════════════════════════════

    def log_action_distribution(self, batch: dict, global_step: int):
        """Log action frequency distribution from the replay batch."""
        if not self.enabled:
            return

        try:
            actions = batch.get("action")
            if actions is None:
                return

            if isinstance(actions, torch.Tensor):
                actions = actions.detach().cpu().numpy()

            # For one-hot actions, find the argmax
            if actions.ndim >= 2 and actions.shape[-1] > 1:
                flat = actions.reshape(-1, actions.shape[-1])
                action_ids = flat.argmax(axis=-1)
                self._wandb.log({
                    "actions/distribution": self._wandb.Histogram(action_ids),
                    "actions/unique_count": len(np.unique(action_ids)),
                    "actions/most_common": int(np.bincount(action_ids).argmax()),
                }, step=global_step)

        except Exception as e:
            print(f"[WandB] Action distribution failed: {e}")

    # ═══════════════════════════════════════════
    #  MODEL SUMMARY TABLE
    # ═══════════════════════════════════════════

    def log_model_summary(self, model):
        """Log architecture details as a WandB table."""
        if not self.enabled:
            return

        data = []
        for name, module in [
            ("Encoder", model.encoder_network),
            ("Decoder", model.decoder_network),
            ("TSSM", model.dynamics_model),
            ("Policy", model.policy_network),
            ("Value", model.value_network),
            ("Reward", model.reward_network),
            ("Continue", model.continue_network),
            ("CPC", model.contrastive_network),
        ]:
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            data.append([name, total, trainable, total * 4 / 1e6])  # MB assuming float32

        table = self._wandb.Table(
            columns=["Component", "Parameters", "Trainable", "Size (MB)"],
            data=data,
        )
        self._wandb.log({"model/architecture": table})

        # Encoder branch details
        if hasattr(model.encoder_network, "encoders"):
            branch_data = []
            for bname, enc in model.encoder_network.encoders.items():
                p = sum(x.numel() for x in enc.parameters())
                branch_data.append([
                    bname,
                    f"{enc.dim_input_cnn}ch × {enc.image_size[0]}×{enc.image_size[1]}",
                    enc.dim_concat,
                    p,
                ])
            branch_table = self._wandb.Table(
                columns=["Branch", "Input Shape", "Output Dim", "Params"],
                data=branch_data,
            )
            self._wandb.log({"model/encoder_branches": branch_table})

    # ═══════════════════════════════════════════
    #  LIFECYCLE
    # ═══════════════════════════════════════════

    def finish(self):
        if self.enabled and self._run:
            self._wandb.finish()
