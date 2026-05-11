"""
Rich Console Logger — Beautiful terminal output for WMAgent training.
Single Responsibility: All Rich console formatting lives here.
"""
import datetime
import torch
import numpy as np
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeRemainingColumn,
    SpinnerColumn, MofNCompleteColumn,
)
from rich import box

from torch_wm.utils import get_console
from torch_wm.utils.logger import print_banner


class RichLogger:
    """Handles all Rich console output: banners, tables, progress bars.
    Uses the global shared Console from torch_wm.utils to enforce DRY.
    """

    def __init__(self):
        self.console = get_console()

    # ── Banner ──
    def print_banner(self):
        print_banner()

    # ── Hardware ──
    def print_hardware(self):
        t = Table(title="[bold green]🖥️  Hardware[/bold green]", box=box.DOUBLE_EDGE)
        t.add_column("Property", style="cyan", width=22)
        t.add_column("Value", style="green", width=42)
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_properties(0)
            t.add_row("GPU", torch.cuda.get_device_name(0))
            t.add_row("VRAM", f"{gpu.total_memory / 1e9:.1f} GB")
            t.add_row("CUDA", torch.version.cuda or "N/A")
            t.add_row("cuDNN", str(torch.backends.cudnn.version()))
        else:
            t.add_row("Device", "CPU")
        t.add_row("PyTorch", torch.__version__)
        self.console.print(t)

    # ── Config ──
    def print_config(self, config):
        t = Table(title="[bold yellow]⚙️  Configuration[/bold yellow]", box=box.DOUBLE_EDGE)
        t.add_column("Parameter", style="cyan", width=22)
        t.add_column("Value", style="white", width=15)
        t.add_column("Description", style="dim", width=28)
        rows = [
            ("batch_size", config.batch_size, "Sequences per batch"),
            ("batch_length", config.batch_length, "Timesteps per sequence"),
            ("horizon", config.horizon, "Imagination rollout"),
            ("model_lr", f"{float(config.model_lr):.1e}", "World model LR"),
            ("actor_lr", f"{float(config.actor_lr):.1e}", "Policy LR"),
            ("critic_lr", f"{float(config.critic_lr):.1e}", "Value LR"),
            ("grad_clip", config.grad_clip, "Gradient clip norm"),
            ("stoch_size", config.stoch_size, "Stochastic dim"),
            ("discrete", config.discrete, "Discrete categories"),
            ("hidden_size", config.hidden_size, "Transformer hidden"),
            ("num_blocks", config.num_blocks_trans, "Transformer blocks"),
            ("context_window", config.att_context_left, "Attention context"),
            ("precision", config.precision, "Compute precision"),
        ]
        for name, val, desc in rows:
            t.add_row(name, str(val), desc)
        self.console.print(t)

    # ── Model Architecture ──
    def print_architecture(self, model):
        t = Table(title="[bold magenta]🧠 Architecture[/bold magenta]", box=box.DOUBLE_EDGE)
        t.add_column("Component", style="cyan", width=28)
        t.add_column("Params", style="green", justify="right", width=12)
        t.add_column("Details", style="dim", width=32)

        components = [
            ("Encoder (Multi-Branch)", model.encoder_network,
             f"Branches: {list(model.encoder_network.encoders.keys())}"),
            ("Decoder (Multi-Branch)", model.decoder_network, "Feat → Image"),
            ("TSSM", model.dynamics_model, f"Blocks: {model.config.num_blocks_trans}"),
            ("Policy", model.policy_network, f"Actions: {model.dynamics_model.num_actions}"),
            ("Value", model.value_network, "Critic head"),
            ("Reward", model.reward_network, "Reward prediction"),
            ("Continue", model.continue_network, "Discount prediction"),
            ("CPC", model.contrastive_network, f"Heads: {len(model.contrastive_network)}"),
            ("Loss Manager (JEPA)", model.world_model.loss_manager, "Learnable Projectors"),
        ]
        total = 0
        for name, mod, det in components:
            p = sum(x.numel() for x in mod.parameters())
            total += p
            t.add_row(name, f"{p:,}", det)
        t.add_section()
        t.add_row("[bold]TOTAL[/bold]", f"[bold]{total:,}[/bold]", "")
        self.console.print(t)

        # Encoder branches
        et = Table(title="[bold blue]📡 Encoder Branches[/bold blue]", box=box.ROUNDED)
        et.add_column("Branch", style="cyan")
        et.add_column("Role", style="magenta")
        et.add_column("Input", style="green")
        et.add_column("Output Dim", style="yellow")
        et.add_column("Params", style="white", justify="right")
        tssm_set = set(getattr(model.encoder_network, '_tssm_branches', []))
        for name, enc in model.encoder_network.encoders.items():
            inp = f"{enc.dim_input_cnn}ch × {enc.image_size[0]}×{enc.image_size[1]}"
            p = sum(x.numel() for x in enc.parameters())
            role = "🧠 TSSM" if name in tssm_set else "📡 Signal"
            et.add_row(name, role, inp, str(enc.dim_concat), f"{p:,}")
        et.add_section()
        et.add_row("[bold]TSSM[/bold]", "", "", f"[bold]{model.encoder_network.dim_concat}[/bold]", "")
        if model.encoder_network.dim_signal > 0:
            et.add_row("[bold]Signal[/bold]", "", "", f"[bold]{model.encoder_network.dim_signal}[/bold]", "")
        self.console.print(et)

    # ── Loss Config ──
    def print_losses(self, config):
        t = Table(title="[bold red]📊 Loss Functions[/bold red]", box=box.DOUBLE_EDGE)
        t.add_column("Loss", style="cyan", width=18)
        t.add_column("Enabled", width=8)
        t.add_column("Weight", style="yellow", width=8)
        modules = config.get("modules", {}).get("losses", {})
        for name, cfg in modules.items():
            on = "✅" if cfg.get("enabled", False) else "❌"
            t.add_row(name.upper(), on, str(cfg.get("weight", 0.0)))
        self.console.print(t)

    # ── Replay Info ──
    def print_replay(self, replay, sample):
        t = Table(title="[bold green]💾 Replay Buffer[/bold green]", box=box.DOUBLE_EDGE)
        t.add_column("Key", style="cyan", width=18)
        t.add_column("Shape / Info", style="white", width=48)
        t.add_row("[bold]Total Steps[/bold]", f"[bold]{len(replay):,}[/bold]")
        t.add_section()
        roles = {
            "camera": "🎥 RGB (World Model)",
            "birdeye_wpt": "🗺️  BEV (Navigation)",
            "action": "🎮 Discrete Action",
            "reward": "⭐ Reward Signal",
            "collision": "💥 Collision",
        }
        for k, v in sample.items():
            role = roles.get(k, "")
            t.add_row(k, f"{v.shape}  {v.dtype}  {role}")
        self.console.print(t)

    # ── Training Plan ──
    def print_plan(self, steps_per_epoch, num_epochs, batch_size, batch_length, horizon):
        bs = batch_size * batch_length
        self.console.print(Panel(
            f"[bold]Steps/epoch: [cyan]{steps_per_epoch}[/cyan]  |  "
            f"Epochs: [cyan]{num_epochs}[/cyan]  |  "
            f"Batch: [cyan]{batch_size}×{batch_length}={bs}[/cyan]  |  "
            f"Horizon: [cyan]{horizon}[/cyan]  |  "
            f"Total iters: [cyan]{steps_per_epoch * num_epochs:,}[/cyan]",
            title="[bold yellow]🚀 Training Plan[/bold yellow]",
            border_style="yellow",
        ))

    # ── Epoch Progress Bar (context manager) ──
    def epoch_progress(self, epoch, total_epochs, total_steps):
        return Progress(
            SpinnerColumn(),
            TextColumn(f"[bold blue]Epoch {epoch}/{total_epochs}"),
            BarColumn(bar_width=10),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            TextColumn("[dim]{task.fields[status]}"),
            console=self.console,
            transient=True,
        )

    def print_epoch(self, epoch, total_epochs, metrics, elapsed_str, global_step):
        """Single-line compact epoch summary — no tables, no truncation."""
        vals = {}
        for k, v in metrics.items():
            vals[k] = v.item() if isinstance(v, torch.Tensor) else float(v)

        # Header
        self.console.print(
            f"[bold]📈 E{epoch}/{total_epochs}[/bold] "
            f"[dim]step={global_step:,} t={elapsed_str}[/dim]  "
            f"[yellow]wm={vals.get('wm_loss', 0):.3f}[/yellow]  "
            f"[green]actor={vals.get('actor_loss', 0):.3f}[/green]  "
            f"[blue]critic={vals.get('critic_loss', 0):.3f}[/blue]  "
            f"[dim]│[/dim]  "
            f"[red]kl[/red]={vals.get('loss_kl', 0):.3f}  "
            f"[red]rec[/red]={vals.get('loss_reconstruction', 0):.4f}  "
            f"[red]rew[/red]={vals.get('loss_reward', 0):.3f}  "
            f"[red]cpc[/red]={vals.get('loss_cpc', 0):.3f}  "
            f"[red]curv[/red]={vals.get('loss_curvature', 0):.3f}  "
            f"[red]disc[/red]={vals.get('loss_discount', 0):.4f}  "
            f"[red]sig[/red]={vals.get('loss_sigreg', 0):.2f}  "
            f"[red]je[/red]={vals.get('loss_je_sim', 0):.3f}  "
            f"[cyan]cpc_acc[/cyan]={vals.get('cpc/cpc_accuracy', 0):.4f}"
        )

    # ── GPU Status ──
    def print_gpu_status(self, speed, epoch_time):
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.console.print(
                f"  [dim]⚡ {speed:.1f} it/s  |  "
                f"🔋 VRAM: {mem:.1f}/{total:.1f} GB  |  "
                f"⏱️  {epoch_time:.1f}s[/dim]"
            )

    # ── Checkpoint ──
    def print_save(self, msg):
        self.console.print(f"  [bold green]💾 {msg}[/bold green]")

    # ── Final Summary ──
    def print_summary(self, global_step, best_loss, final_path, tb_dir):
        self.console.print(Panel(
            f"[bold green]✅ Training Complete![/bold green]\n\n"
            f"  Total Steps: [cyan]{global_step:,}[/cyan]\n"
            f"  Best WM Loss: [cyan]{best_loss:.4f}[/cyan]\n"
            f"  Model: [cyan]{final_path}[/cyan]\n"
            f"  TensorBoard: [cyan]{tb_dir}[/cyan]",
            title="[bold]Summary[/bold]", border_style="green",
        ))

    def print_error(self, msg):
        self.console.print(f"[bold red]❌ {msg}[/bold red]")

    def print_warning(self, msg):
        self.console.print(f"[yellow]⚠️  {msg}[/yellow]")

    def print_ok(self, msg):
        self.console.print(f"[bold green]✅ {msg}[/bold green]")
