"""
WMAgent Offline Training Pipeline
Trains the dual-encoder world model from pre-collected expert data.

Usage:
    uv run python -m torch_wm.rl.train_offline
    uv run python -m torch_wm.rl.train_offline --replay_dir ./expert_data/replay
"""
import pathlib
import sys
import warnings
import time

from torch_wm.rl._setup_path import setup; setup()

import torch
import numpy as np
import embodied
import ruamel.yaml as yaml

from torch_wm.rl.agent import WorldModelAgent
from torch_wm.rl.loggers import LogComposer

warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")


# ═══════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════

def load_replay(replay_dir: pathlib.Path, config):
    """Load and validate the replay buffer."""
    replay = embodied.replay.Uniform(
        config.batch_length,
        int(float(config.replay_size)),
        replay_dir,
    )
    return replay


def build_spaces(sample: dict):
    """Infer obs/act spaces from a single replay sample."""
    obs_space = {}
    for k, v in sample.items():
        if k in ("action", "reward", "is_first", "is_last", "is_terminal", "id"):
            continue
        obs_space[k] = embodied.Space(dtype=str(v.dtype), shape=v.shape)

    act_space = {}
    if "action" in sample:
        act_space["action"] = embodied.Space(
            dtype=str(sample["action"].dtype), shape=sample["action"].shape,
        )
    return obs_space, act_space


# ═══════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════

def train_offline(agent, replay, config, logger: LogComposer):
    """Core offline training loop — thin orchestration only."""
    dataset = agent.dataset(replay.dataset)

    # Compute schedule
    batch_steps = config.batch_size * config.batch_length
    steps_per_epoch = max(1, len(replay) // batch_steps)
    num_epochs = min(
        max(1, int(float(config.steps)) // (steps_per_epoch * batch_steps)),
        500,
    )
    logger.on_train_plan(steps_per_epoch, num_epochs)

    state = None
    global_step = 0
    best_wm_loss = float("inf")
    last_batch = None

    try:
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            epoch_acc = {}

            with logger.epoch_progress(epoch, num_epochs, steps_per_epoch) as progress:
                task = progress.add_task("Training", total=steps_per_epoch, status="...")

                for _ in range(steps_per_epoch):
                    batch = next(dataset)
                    outs, state, metrics = agent.train(batch, state)
                    global_step += 1
                    last_batch = batch

                    # Accumulate
                    for k, v in metrics.items():
                        val = v.item() if isinstance(v, torch.Tensor) else float(v)
                        epoch_acc.setdefault(k, []).append(val)

                    # Per-step logging (delegated to composer)
                    logger.on_step(metrics, global_step)

                    # Progress bar
                    m = metrics
                    def _get_val(k):
                        v = m.get(k, 0)
                        return v.item() if isinstance(v, torch.Tensor) else float(v)

                    wm = _get_val("wm_loss")
                    rec = _get_val("loss_reconstruction")
                    kl = _get_val("loss_kl")
                    rew = _get_val("loss_reward")
                    cpc = _get_val("loss_cpc")
                    curv = _get_val("loss_curvature")
                    sig = _get_val("loss_sigreg")
                    je = _get_val("loss_je_sim")
                    disc = _get_val("loss_discount")
                    act = _get_val("actor_loss")
                    crit = _get_val("critic_loss")
                    
                    # Trích xuất GPU VRAM đang dùng (GB)
                    vram_gb = torch.cuda.memory_reserved(0) / (1024 ** 3) if torch.cuda.is_available() else 0.0
                        
                    status_str = (
                        f"[yellow]wm:{wm:.0f}[/yellow] "
                        f"| [cyan]rec:{rec:.0f}[/cyan] "
                        f"| [magenta]kl:{kl:.1f}[/magenta] "
                        f"| [green]rew:{rew:.2f}[/green] "
                        f"| [blue]cpc:{cpc:.2f}[/blue] "
                        f"| [white]curv:{curv:.2f}[/white] "
                        f"| [red]sig:{sig:.2f}[/red] "
                        f"| [bold magenta]je:{je:.2f}[/bold magenta] "
                        f"| [yellow]disc:{disc:.2f}[/yellow] "
                        f"| [bold cyan]act:{act:.2f}[/bold cyan] "
                        f"| [bold red]crit:{crit:.2f}[/bold red] "
                        f"| [bold green]🎮 {vram_gb:.1f}GB[/bold green]"
                    )
                    progress.update(task, advance=1, status=status_str)
                    
                    # --- LOCAL CONTINUOUS RECONSTRUCTION SNAPSHOT FOR USER ---
                    if global_step % 50 == 0:
                        try:
                            import torchvision
                            with torch.no_grad():
                                # Generate small snapshot for the user locally
                                posts = agent.model._last_outputs.get("posts")
                                states_rec = agent.model._last_outputs.get("states_rec_dist")
                                
                                if posts is not None and states_rec is not None:
                                    if not isinstance(states_rec, dict): states_rec = {"camera": states_rec}
                                    if "camera" in states_rec:
                                        # (B*L, C, H, W)
                                        rec_dist = states_rec["camera"]
                                        rec_frames = rec_dist.mode() if hasattr(rec_dist, 'mode') else rec_dist.mean()
                                        
                                        # Get GT from batch
                                        raw_batch_obs = agent._preprocess_obs(batch)
                                        raw_cam = raw_batch_obs.get("camera")
                                        if raw_cam is not None:
                                            # Take first 8 frames from the first sequence in the batch
                                            N = min(8, raw_cam.shape[1])
                                            raw_seq = ((raw_cam[0, :N] + 0.5).clamp(0, 1)).cpu()
                                            
                                            # Align rec frames logic
                                            B, L = raw_cam.shape[0], raw_cam.shape[1]
                                            C, H, W = rec_frames.shape[-3], rec_frames.shape[-2], rec_frames.shape[-1]
                                            rec_seq = ((rec_frames.view(B, L, C, H, W)[0, :N] + 0.5).clamp(0, 1)).cpu()
                                            
                                            # Concat GT (top) and Rec (bottom)
                                            # Create grid
                                            grid = torchvision.utils.make_grid(torch.cat([raw_seq, rec_seq], dim=0), nrow=N)
                                            torchvision.utils.save_image(grid, "latest_recon.jpg")
                        except Exception as e:
                            with open("recon_error.txt", "w") as f:
                                f.write("Extract recon failed: " + str(e))

            # Epoch summary
            epoch_time = time.time() - epoch_start
            avg_metrics = {k: np.mean(v) for k, v in epoch_acc.items()}

            # Preprocess batch for reconstruction logging (need to match agent's format)
            preprocessed_batch = None
            if last_batch is not None:
                preprocessed_batch = agent._preprocess_obs(last_batch)

            logger.on_epoch_end(
                epoch, num_epochs, avg_metrics, global_step,
                epoch_time, steps_per_epoch,
                model=agent.model, batch=preprocessed_batch,
            )

            # Save best
            avg_wm = avg_metrics.get("wm_loss", avg_metrics.get("train/wm_loss", float("inf")))
            if avg_wm < best_wm_loss:
                best_wm_loss = avg_wm
                torch.save({
                    "model": agent.model.state_dict(),
                    "epoch": epoch, "global_step": global_step, "wm_loss": avg_wm,
                }, str(logger.logdir / "best_model.pt"))
                logger.on_save(f"Best model (wm_loss: {avg_wm:.4f})")

            # Periodic checkpoint
            if epoch % 10 == 0:
                torch.save({
                    "model": agent.model.state_dict(),
                    "epoch": epoch, "global_step": global_step,
                }, str(logger.logdir / f"checkpoint_epoch_{epoch}.pt"))
                logger.on_save(f"Checkpoint epoch_{epoch}")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
    except Exception:
        import traceback
        traceback.print_exc()

    # Final save
    final_path = str(logger.logdir / "final_model.pt")
    torch.save({
        "model": agent.model.state_dict(),
        "global_step": global_step, "wm_loss": best_wm_loss,
    }, final_path)

    logger.on_train_end(global_step, best_wm_loss, final_path)


# ═══════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if "--help" in argv:
        print("Usage: train_offline.py")
        temp_flags = embodied.Flags(method="dreamerv3")
        parsed_temp, remaining = temp_flags.parse_known(argv)
        config_path = pathlib.Path(__file__).resolve().parent / "config" / f"{parsed_temp.method}.yaml"
        with open(config_path, "r") as f:
            model_configs = yaml.YAML(typ="safe").load(f)
        config = embodied.Config(model_configs["defaults"])
        embodied.Flags(config).parse(remaining)

    # Initial flags to determine which config file to load
    temp_flags = embodied.Flags(method="dreamerv3")
    parsed_temp, remaining = temp_flags.parse_known(argv)
    
    # Config
    config_path = pathlib.Path(__file__).resolve().parent / "config" / f"{parsed_temp.method}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, "r") as f:
        model_configs = yaml.YAML(typ="safe").load(f)
    
    config = embodied.Config(model_configs["defaults"])
    parsed, other = embodied.Flags(replay_dir="./expert_data/replay").parse_known(remaining)
    config = embodied.Flags(config).parse(other)

    # Logdir
    logdir = pathlib.Path(config.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    # Logger (composite)
    logger = LogComposer(config, logdir)

    # Replay
    replay = load_replay(pathlib.Path(parsed.replay_dir), config)
    min_data = config.batch_length * config.batch_size
    if len(replay) < min_data:
        logger.error(f"Not enough data! Need >= {min_data}, got {len(replay)}")
        return

    sample = next(replay.dataset())
    obs_space, act_space = build_spaces(sample)

    # Agent
    step = embodied.Counter()
    agent = WorldModelAgent(obs_space, act_space, step, config)

    # Startup info
    logger.on_train_start(agent.model, replay, sample)

    # Train
    train_offline(agent, replay, config, logger)


if __name__ == "__main__":
    main()
