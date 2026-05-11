"""
TensorBoard Logger — Scalars, histograms, and system metrics.
Single Responsibility: All TensorBoard writes live here.
"""
import torch
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """Writes scalars, histograms, and images to TensorBoard."""

    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_step(self, metrics: dict, global_step: int):
        """Log per-step scalar metrics."""
        for k, v in metrics.items():
            val = v.item() if isinstance(v, torch.Tensor) else float(v)
            self.writer.add_scalar(f"step/{k}", val, global_step)

    def log_epoch(self, metrics: dict, epoch: int, speed: float = 0.0):
        """Log epoch-averaged metrics."""
        for k, v in metrics.items():
            self.writer.add_scalar(f"epoch/{k}", float(v), epoch)
        if speed > 0:
            self.writer.add_scalar("epoch/speed_it_s", speed, epoch)

    def log_system(self, global_step: int):
        """Log GPU memory usage."""
        if torch.cuda.is_available():
            mem_gb = torch.cuda.memory_allocated() / 1e9
            self.writer.add_scalar("system/gpu_memory_gb", mem_gb, global_step)

    def log_histograms(self, model, global_step: int):
        """Log weight and gradient histograms for key components."""
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Only key modules to avoid flooding
                for prefix in ("encoder_network", "tssm", "policy_network", "decoder_network"):
                    if name.startswith(prefix):
                        safe_name = name.replace(".", "/")
                        self.writer.add_histogram(f"weights/{safe_name}", param.data, global_step)
                        self.writer.add_histogram(f"grads/{safe_name}", param.grad.data, global_step)
                        break

    def log_images(self, tag: str, images: torch.Tensor, global_step: int):
        """Log a batch of images as a grid. Images: (N, C, H, W) in [0, 1]."""
        try:
            import torchvision
            grid = torchvision.utils.make_grid(images.clamp(0, 1), nrow=8, padding=2)
            self.writer.add_image(tag, grid, global_step)
        except ImportError:
            pass

    def close(self):
        self.writer.close()
