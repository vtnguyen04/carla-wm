"""
Log Composer — Orchestrates all loggers with a unified lifecycle API.
Single Responsibility: Routes log calls to Rich, TensorBoard, and WandB.
Open/Closed: New loggers can be added without modifying existing ones.
"""
import time
import datetime
import pathlib
import torch
import numpy as np

from .rich_logger import RichLogger
from .tb_logger import TensorBoardLogger
from .wandb_logger import WandBLogger


class LogComposer:
    """
    Composite logger that delegates to Rich, TensorBoard, and WandB.

    Lifecycle hooks:
        on_train_start  → Print system info, architecture, log model summary
        on_train_plan   → Show training schedule
        on_step         → Per-step scalar logging (throttled)
        on_epoch_end    → Full epoch summary + periodic deep logging
        on_train_end    → Final summary + cleanup
    """

    def __init__(self, config, logdir: pathlib.Path):
        self.config = config
        self.logdir = logdir
        self.start_time = time.time()

        # Sub-loggers (Dependency Injection)
        self.rich = RichLogger()
        self.tb = TensorBoardLogger(str(logdir / "tensorboard"))
        project = config.get("wandb_project", "twister-carla")
        self.wandb = WandBLogger(config=dict(config), log_dir=str(logdir), project=project)

        # Logging intervals (configurable)
        self._step_log_interval = 50       # Scalars every N steps
        self._recon_log_interval = 1       # Reconstruction every epoch
        self._histogram_interval = 10      # Weight histograms every N epochs
        self._gradient_interval = 5        # Gradient norms every N epochs

    # ──────────────────────────────────────────
    #  LIFECYCLE: Train Start
    # ──────────────────────────────────────────

    def on_train_start(self, model, replay, sample):
        """Called once before training begins."""
        self.rich.print_banner()
        self.rich.print_hardware()
        self.rich.print_config(self.config)
        self.rich.print_losses(self.config)
        self.rich.print_replay(replay, sample)
        self.rich.print_architecture(model)

        # WandB model info
        self.wandb.log_model_summary(model)

    def on_train_plan(self, steps_per_epoch, num_epochs):
        """Called after computing the training plan."""
        self.rich.print_plan(
            steps_per_epoch, num_epochs,
            self.config.batch_size, self.config.batch_length,
            self.config.horizon,
        )

    # ──────────────────────────────────────────
    #  LIFECYCLE: Per-Step
    # ──────────────────────────────────────────

    def on_step(self, metrics: dict, global_step: int):
        """Periodic scalar logging to TB + WandB (throttled)."""
        if global_step % self._step_log_interval == 0:
            self.tb.log_step(metrics, global_step)
            self.tb.log_system(global_step)
            self.wandb.log_step(metrics, global_step)

    # ──────────────────────────────────────────
    #  LIFECYCLE: Epoch Progress
    # ──────────────────────────────────────────

    def epoch_progress(self, epoch, total_epochs, steps):
        """Returns a Rich progress context manager."""
        return self.rich.epoch_progress(epoch, total_epochs, steps)

    # ──────────────────────────────────────────
    #  LIFECYCLE: Epoch End
    # ──────────────────────────────────────────

    def on_epoch_end(self, epoch: int, total_epochs: int,
                     avg_metrics: dict, global_step: int,
                     epoch_time: float, steps_per_epoch: int,
                     model=None, batch=None):
        """Full epoch logging: console, scalars, and periodic deep analytics."""
        elapsed = str(datetime.timedelta(seconds=int(time.time() - self.start_time)))
        speed = steps_per_epoch / max(epoch_time, 0.01)

        # ── Console ──
        self.rich.print_epoch(epoch, total_epochs, avg_metrics, elapsed, global_step)
        self.rich.print_gpu_status(speed, epoch_time)

        # ── TensorBoard epoch scalars ──
        self.tb.log_epoch(avg_metrics, epoch, speed)

        # ── WandB epoch scalars ──
        gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        self.wandb.log_epoch(avg_metrics, epoch, global_step, speed, gpu_mem)

        # Periodic deep logging (requires model + batch) ──
        if model is not None and batch is not None:
            self._deep_log(epoch, global_step, model, batch)
        elif model is not None:
            # Case for imagination only logging if batch is not available
            if epoch % self._recon_log_interval == 0:
                self.wandb.log_imagination(model, global_step, num_steps=self.config.get("H", 15))

    def _deep_log(self, epoch, global_step, model, batch):
        """Periodic expensive logging: reconstructions, gradients, latents."""
        # Reconstruction video + images (every N epochs)
        if epoch % self._recon_log_interval == 0:
            self.wandb.log_reconstructions(model, batch, global_step)
            self.wandb.log_imagination(model, global_step, num_steps=self.config.get("H", 15))

            # Latent space analytics

            if hasattr(model, "_last_outputs") and model._last_outputs:
                self.wandb.log_latent_stats(model._last_outputs, global_step)
                self.wandb.log_reward_analysis(batch, model._last_outputs, global_step)

            # Action distribution
            self.wandb.log_action_distribution(batch, global_step)

        # Gradient norms (every N epochs)
        if epoch % self._gradient_interval == 0:
            self.wandb.log_gradient_norms(model, global_step)

        # Weight histograms to TensorBoard (every N epochs)
        if epoch % self._histogram_interval == 0:
            self.tb.log_histograms(model, global_step)

    # ──────────────────────────────────────────
    #  LIFECYCLE: Checkpointing
    # ──────────────────────────────────────────

    def on_save(self, msg: str):
        self.rich.print_save(msg)

    # ──────────────────────────────────────────
    #  LIFECYCLE: Train End
    # ──────────────────────────────────────────

    def on_train_end(self, global_step: int, best_loss: float, final_path: str):
        self.rich.print_summary(
            global_step, best_loss, final_path,
            str(self.logdir / "tensorboard"),
        )
        self.tb.close()
        self.wandb.finish()

    # ──────────────────────────────────────────
    #  UTILITIES
    # ──────────────────────────────────────────

    def error(self, msg: str):
        self.rich.print_error(msg)

    def warning(self, msg: str):
        self.rich.print_warning(msg)

    def ok(self, msg: str):
        self.rich.print_ok(msg)
