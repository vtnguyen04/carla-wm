"""Module Manager for WMAgent-CARLA.

Manages the lifecycle of active modules based on configuration.
Handles both dict and AttrDict configs.
"""

from typing import Any, Dict, List

import torch
import torch.nn as nn

from .base import BaseLoss
from .registry import ModuleRegistry

def get_config_value(config, *keys, default=None):
    """Safely get nested value from dict or AttrDict."""
    val = config
    for key in keys:
        if hasattr(val, 'get'):
            val = val.get(key, default)
        elif hasattr(val, '__getitem__'):
            try:
                val = val[key]
            except (KeyError, IndexError):
                return default
        else:
            val = getattr(val, key, default)
        if val is default:
            return default
    return val
def ensure_losses_registered():
    """Ensure all loss modules are registered in ModuleRegistry."""
    # Import all loss modules explicitly to trigger @register decorators
    try:
        import torch_wm.modules.losses.kl_loss
        import torch_wm.modules.losses.reward_loss
        import torch_wm.modules.losses.discount_loss
        import torch_wm.modules.losses.reconstruction_loss
        import torch_wm.modules.losses.cpc_loss
        import torch_wm.modules.losses.kl_balancing_loss
        import torch_wm.modules.losses.joint_embedding_loss
        import torch_wm.modules.losses.vcreg_loss
        
        # Import Regularizers
        import torch_wm.modules.regularizers.curvature
        import torch_wm.modules.regularizers.sigreg
        import torch_wm.modules.regularizers.straightening
        import torch_wm.modules.regularizers.vcreg
    except ImportError as e:
        from torch_wm.utils import get_logger
        logger = get_logger(__name__)
        logger.warning(f"Some loss modules could not be imported: {e}")


class LossManager(nn.Module):
    """Manages active loss modules.

    Reads config, instantiates only enabled losses,
    and aggregates them during training.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.active_losses = nn.ModuleList()
        # Ensure losses are registered before building
        ensure_losses_registered()
        self._build_losses()

    def _build_losses(self):
        """Instantiate losses and regularizers that are enabled in config."""
        losses_config = get_config_value(self.config, "modules", "losses", default={}) or {}
        reg_config = get_config_value(self.config, "modules", "regularizers", default={}) or {}
        
        from torch_wm.utils import get_console
        console = get_console()
        from rich.panel import Panel
        from rich.table import Table
        
        table = Table(show_header=True, header_style="bold cyan", border_style="dim", box=None)
        table.add_column("Type", style="magenta")
        table.add_column("Status", justify="center")
        table.add_column("Module Name", style="bold")
        table.add_column("Weight", justify="right")

        # Combine configs but keep track of type for display
        all_modules = [("Loss", name, losses_config.get(name, {})) for name in losses_config.keys()]
        all_modules += [("Regularizer", name, reg_config.get(name, {})) for name in reg_config.keys()]
        
        # If no config provided at all, fallback to registry keys
        if not all_modules:
            all_modules = [("Module", name, {}) for name in ModuleRegistry._registry.keys()]

        # Filter out duplicates (if any) and only iterate over what's actually registered
        processed = set()
        for mod_type, name, loss_cfg in all_modules:
            if name in processed or name not in ModuleRegistry._registry:
                continue
            processed.add(name)
            
            loss_cls = ModuleRegistry._registry[name]
            enabled = loss_cfg.get("enabled", False) if isinstance(loss_cfg, dict) else getattr(loss_cfg, "enabled", False)
            
            if enabled:
                weight = loss_cfg.get("weight", 1.0) if isinstance(loss_cfg, dict) else getattr(loss_cfg, "weight", 1.0)
                loss = loss_cls(config=loss_cfg, weight=weight)
                self.active_losses.append(loss)
                table.add_row(mod_type, "[green]✔ Active[/green]", name, f"w={weight}")
            else:
                table.add_row(mod_type, "[dim]Skipped[/dim]", f"[dim]{name}[/dim]", "-")
        
        console.print()
        if not self.active_losses:
            console.print(Panel(table, title="[yellow]Objective Engine[/yellow]", subtitle="[yellow]No Objectives Activated[/yellow]", border_style="yellow"))
        else:
            console.print(Panel(table, title="[bold green]Objective Engine[/bold green]", subtitle="[bold green]Modules Loaded Successfully[/bold green]", border_style="green"))

    def compute_total_loss(
        self,
        model_outputs: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Compute weighted sum of all active losses.

        Returns:
            Dict with 'total' loss and individual loss values.
        """
        device = 'cpu'
        for v in model_outputs.values():
            if isinstance(v, torch.Tensor):
                device = v.device
                break
            elif isinstance(v, dict):
                first_tensor = next((item for item in v.values() if isinstance(item, torch.Tensor)), None)
                if first_tensor is not None:
                    device = first_tensor.device
                    break

        if not self.active_losses:
            return {"total": torch.tensor(0.0, device=device)}
            
        total_loss = torch.tensor(0.0, device=device)
        losses_dict = {}

        for loss_module in self.active_losses:
            loss_val = loss_module.compute(model_outputs, batch, **kwargs)
            losses_dict[loss_module.name()] = loss_val
            total_loss = total_loss + (loss_module.weight * loss_val)

        losses_dict["total"] = total_loss
        return losses_dict

    def get_metrics(self) -> Dict[str, Any]:
        """Collect metrics from all active losses."""
        metrics = {}
        for loss in self.active_losses:
            metrics.update(loss.get_metrics())
        return metrics
