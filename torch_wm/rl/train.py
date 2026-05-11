"""
WMAgent Online RL Training Pipeline
The agent interacts with CARLA, collects its own experience, and learns.

Architecture:
    Agent.policy(obs) → action        # Interact with env
    env.step(action)  → obs, reward   # Get reward feedback
    replay.add(transition)            # Store experience
    Agent.train(batch)                # Learn from replay

Usage:
    uv run python -m torch_wm.rl.train
    uv run python -m torch_wm.rl.train --task carla_navigation
"""
import pathlib
import warnings

from torch_wm.rl._setup_path import setup
from embodied.core.wrappers import InfoWrapper; setup()

import embodied
import ruamel.yaml as yaml

import carla_env
from torch_wm.rl.agent import WorldModelAgent
from carla_env.toolkit.utils import get_logger

log = get_logger(log_dir=".", job_name="train")

warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

class FilterObs(embodied.Wrapper):
    """Renames 'action' in obs to 'env_action' to avoid collisions."""
    def __init__(self, env, pattern="action"):
        super().__init__(env)
        self._pattern = pattern

    @property
    def obs_space(self):
        spaces = self.env.obs_space.copy()
        new_spaces = {}
        for k, v in spaces.items():
            if self._pattern in k:
                new_spaces[f"env_{k}"] = v
            else:
                new_spaces[k] = v
        return new_spaces

    def step(self, action):
        obs, info = self.env.step(action)
        new_obs = {}
        for k, v in obs.items():
            if self._pattern in k:
                new_obs[f"env_{k}"] = v
            else:
                new_obs[k] = v
        return new_obs, info


def wrap_env(env, config):
    """Apply standard embodied wrappers for RL training."""
    args = config.wrapper
    env = embodied.wrappers.InfoWrapper(env)
    env = FilterObs(env)  # Rename 'action' in obs
    if args.get("repeat", 1) > 1:
        env = embodied.wrappers.ActionRepeat(env, args.repeat)
    for name, space in env.act_space.items():
        if name == "reset":
            continue
        elif space.discrete:
            env = embodied.wrappers.OneHotAction(env, name)
        elif args.get("discretize", 0):
            env = embodied.wrappers.DiscretizeAction(env, name, args.discretize)
        else:
            env = embodied.wrappers.NormalizeAction(env, name)
    env = embodied.wrappers.ExpandScalars(env)
    if args.get("length", 0):
        env = embodied.wrappers.TimeLimit(env, args.length, args.get("reset", True))
    if args.get("checks", False):
        env = embodied.wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.ClipAction(env, name)
    return env

def main(argv=None):
    # Initial flags to determine which config file to load
    temp_flags = embodied.Flags(method="dreamerv3", model_size="defaults")
    temp_parsed, remaining = temp_flags.parse_known(argv)
    
    method = temp_parsed.method
    model_size = temp_parsed.model_size

    config_path = pathlib.Path(__file__).resolve().parent / "config" / f"{method}.yaml"
    if not config_path.exists():
        log.error(f"Config not found: {config_path}")
        return

    model_configs = yaml.YAML(typ="safe").load(config_path.read_text())
    if model_size not in model_configs:
        log.warning(f"Unknown model_size: {model_size}. Falling back to 'defaults'")
        model_size = "defaults"

    config = embodied.Config(model_configs["defaults"])
    if model_size != "defaults":
        config = config.update(model_configs[model_size])

    parsed, other = embodied.Flags(task=["carla_navigation"]).parse_known(remaining)
    env = None
    for name in parsed.task:
        log.info(f"Using task: {name}")
        env, env_config = carla_env.create_task(name, other)
        config = config.update(env_config)

    config = embodied.Flags(config).parse(other)
    log.info(config)

    logdir = embodied.Path(config.logdir)
    step = embodied.Counter()
    outputs = [
        embodied.logger.TerminalOutput(pattern=r".*return$|.*_loss$|.*_lr$"),
        embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
    ]
    if config.get("tensorboard", True):
        outputs.append(embodied.logger.TensorBoardOutput(logdir))

    if config.get("wandb", True):
        outputs.append(embodied.logger.WandBOutput(
            run_name=f"{config.task}-{pathlib.Path(config.logdir).name}",
            config=config,
            project=config.get("wandb_project", "world-model-carla"),
        ))
    logger = embodied.Logger(step, outputs)

    # ── Environment ──
    from embodied.envs import from_gym
    if env is not None:
        env = from_gym.FromGym(env)
        env = wrap_env(env, config)
        env = embodied.BatchEnv([env], parallel=False)

        cleanup = [env]
        try:
            # ── Agent ──
            agent = WorldModelAgent(env.obs_space, env.act_space, step, config)

            # ── Replay Buffer (online — agent fills it) ──
            replay = embodied.replay.Uniform(
                config.batch_length, int(float(config.get("replay_size", 0))), logdir / "replay"
            )

            batch_size = int(config.get("batch_size", 0))
            batch_length = int(config.get("batch_length", 0))
            steps = int(config.get("steps", 0))
            eval_every = int(config.get("eval_every", 0))
            eval_episodes = int(config.get("eval_episodes", 0))
            log_every = int(config.get("log_every", 0))
            save_every= int(config.get("save_every", 0))


            run_args = embodied.Config(
                **config.run,
                logdir=config.logdir,
                batch_steps=batch_size * batch_length,
                steps=int(float(steps)),
                eval_every=int(float(eval_every)),
                eval_episodes=eval_episodes,
                log_every=int(float(log_every)),
                save_every=int(float(save_every)),
                train_ratio=config.train_ratio,
                train_fill=config.train_fill,
            )
            # Ensure from_checkpoint is present if not in run config
            if "from_checkpoint" not in run_args:
                run_args = run_args.update({"from_checkpoint": config.get("from_checkpoint", "")})

            # ── Online RL Loop ──
            # This is the core:
            #   1. Random exploration → fill replay buffer
            #   2. Agent.policy(obs) → action (interact with CARLA)
            #   3. Env returns reward + next obs
            #   4. Store transition in replay
            #   5. Sample batch from replay → Agent.train(batch)
            #   6. Repeat until done
            embodied.run.train(agent, env, replay, logger, run_args)

        finally:
            for obj in cleanup:
                obj.close()


if __name__ == "__main__":
    main()
