import re
import sys
import warnings
import pathlib
import embodied
import numpy as np
import ruamel.yaml as yaml
from tqdm import tqdm
import carla_env
from torch_wm.rl.agent import WorldModelAgent
from carla_env.toolkit.utils import get_logger

warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")


def wrap_env(env, config):
    args = config.wrapper
    env = embodied.wrappers.InfoWrapper(env)
    if args.get("repeat", 1) > 1:
        env = embodied.wrappers.ActionRepeat(env, args.repeat)
    for name, space in env.act_space.items():
        if name == "reset":
            continue
        elif space.discrete:
            env = embodied.wrappers.OneHotAction(env, name)
        elif args.discretize:
            env = embodied.wrappers.DiscretizeAction(env, name, args.discretize)
        else:
            env = embodied.wrappers.NormalizeAction(env, name)
    env = embodied.wrappers.ExpandScalars(env)
    if args.length:
        env = embodied.wrappers.TimeLimit(env, args.length, args.reset)
    if args.checks:
        env = embodied.wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.ClipAction(env, name)
    return env


def eval_only(agent, env, logger, args):
    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    log = get_logger(log_dir=str(logdir), job_name="eval")
    log.info("Start evaluation.")
    log.info(f"Args: {args}")
    log.info(f"Logdir: {logdir}")
    step = logger.step
    metrics = embodied.Metrics()
    log.info(f"Observation space: {env.obs_space}")
    log.info(f"Action space: {env.act_space}")

    timer = embodied.Timer()
    timer.wrap("agent", agent, ["policy"])
    timer.wrap("env", env, ["step"])
    timer.wrap("logger", logger, ["write"])

    nonzeros = set()

    def per_episode(ep, ep_info):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        logger.add({"length": length, "score": score}, prefix="episode")
        log.info(f"Episode has {length} steps and return {score:.1f}.")
        stats = {}
        for key in args.log_keys_video:
            if key in ep:
                stats[f"policy_{key}"] = ep[key]

        def log_stats(key, value):
            if re.match(args.log_keys_sum, key):
                stats[f"sum_{key}"] = value.sum()
            if re.match(args.log_keys_mean, key):
                stats[f"mean_{key}"] = value.mean()
            if re.match(args.log_keys_max, key):
                stats[f"max_{key}"] = value.max(0).mean()

        for key, value in ep.items():
            if not args.log_zeros and key not in nonzeros and (value == 0).all():
                continue
            nonzeros.add(key)
            log_stats(key, value)
        for key, value in ep_info.items():
            log_stats(key, value)

        logger.add(metrics.result())
        logger.add(timer.stats(), prefix="timer")
        logger.write(fps=True)

        metrics.add(stats, prefix="stats")

    def per_step(tran):
        step.increment()

    driver = embodied.Driver(env)
    driver.on_episode(lambda ep, ep_info, worker: per_episode(ep, ep_info))
    driver.on_step(lambda tran, info, _: per_step(step))

    episode_count = embodied.Counter()  # Thêm bộ đếm episode
    driver.on_episode(
        lambda *args: episode_count.increment()
    )  # Tăng bộ đếm mỗi khi kết thúc episode

    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    if args.from_checkpoint:
        checkpoint.load(args.from_checkpoint, keys=["agent"])
    else:
        raise ValueError("No checkpoint specified.")

    log.info("Start evaluation loop.")
    policy = lambda *args: agent.policy(*args, mode="eval")

    # Progress bar defaults to track Episodes if set, else Steps
    desc = "Episodes" if args.eval_episodes > 0 else "Steps"
    total = args.eval_episodes if args.eval_episodes > 0 else args.steps

    with tqdm(total=total, desc=desc) as pbar:
        prev_ep_count = 0
        while step < args.steps and (
            args.eval_episodes < 0 or episode_count.value < args.eval_episodes
        ):
            driver(policy, steps=100)

            if args.eval_episodes > 0:
                # Update based on completed episodes
                current_ep_count = episode_count.value
                delta = current_ep_count - prev_ep_count
                if delta > 0:
                    pbar.update(delta)
                    prev_ep_count = current_ep_count
            else:
                # Update based on steps
                pbar.update(100)

    logger.write()


def main(argv=None):
    from torch_wm.rl._setup_path import setup; setup()

    # Initial argument parsing
    if argv is None:
        argv = sys.argv[1:]

    temp_flags = embodied.Flags(method="wm_agent", model_size="defaults")
    temp_parsed, temp_other = temp_flags.parse_known(argv)
    method = temp_parsed.method
    model_size = temp_parsed.model_size

    log = get_logger(log_dir=".", job_name="main_eval")

    config_path = embodied.Path(__file__).parent / "config" / "dreamerv3.yaml"

    model_configs = yaml.YAML(typ="safe").load(config_path.read())

    if model_size not in model_configs:
        raise ValueError(f"Unknown model_size: {model_size}. Available: {list(model_configs.keys())}")

    # Initialize Config
    config = embodied.Config({method: model_configs["defaults"]})
    config = config.update({method: model_configs[model_size]})

    # --- Config and Environment Setup ---

    # Parse task name early.
    pre_parsed, other = embodied.Flags(
        task=["carla_navigation"]
    ).parse_known(temp_other)

    task_name = pre_parsed.task[0]
    log.info(f"Using task: {task_name}")

    # If --hires is used, we must inject the video-saving flags into the arguments
    # BEFORE create_task is called, because create_task is what builds the env.
    if "--hires" in other:
        log.info("Hires mode: Injecting video-saving flags for create_task.")
        video_flags = [
            "--env.display.enable=True",
            "--env.display.save_video=True",
            "--env.display.hires=True",
        ]
        other = video_flags + other

    # Create the environment using the method from the training script.
    # It will now see the injected video flags if --hires was present.
    env, env_config = carla_env.create_task(task_name, other)
    config = config.update(env_config)

    # Load and merge the full hires config for the rest of the script (agent, logger).
    # The original --hires flag is still in `other` to trigger this.
    if "--hires" in other:
        log.info("High-resolution mode: Loading full hires config for agent/logger.")
        hires_path = embodied.Path(__file__).parent / "eval_hires.yaml"
        hires_config = yaml.YAML(typ="safe").load(hires_path.read())
        config = config.update(hires_config)
        # Clean the --hires flag so it's not parsed again later.
        other = [arg for arg in other if arg != "--hires"]

    # Filter arguments to prevent mismatch errors for the agent config.
    if method == "tdmpc2":
        other = [arg for arg in other if not arg.startswith("--dreamerv3.")]
    elif method == "dreamerv3":
        other = [arg for arg in other if not arg.startswith("--tdmpc2.")]

    # Inject runtime defaults early so flags can override them
    runtime_defaults = {
        "run.log_keys_sum": "(travel_distance|destination_reached|out_of_lane|time_exceeded|is_collision|timesteps)",
        "run.log_keys_mean": "(travel_distance|ttc|speed_norm|wpt_dis)",
        "run.log_keys_max": "(travel_distance|ttc|speed_norm|wpt_dis)",
        "run.steps": 5e4,
        "run.eval_episodes": -1,
    }
    config = config.update({method: runtime_defaults})

    # Final parse of all flags. This is necessary for the agent and run configs.
    config = embodied.Flags(config).parse(other)

    # --- Logdir and Logger Setup ---
    method_config = config[method]
    logdir_str = method_config.logdir
    if logdir_str == "/dev/null":
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir_str = f"eval_logs/{task_name}_{timestamp}"
        print(f"Logdir explicitly set to {logdir_str} to avoid /dev/null error")

    logdir = embodied.Path(logdir_str)
    step = embodied.Counter()
    
    # Unified Logger Outputs
    outputs = [
        embodied.logger.TerminalOutput(pattern=r".*reward.*|.*return.*|.*loss.*"),
        embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
        embodied.logger.TensorBoardOutput(logdir),
    ]
    
    # Add WandB support for eval tracking
    if method_config.get("wandb", True):
        outputs.append(embodied.logger.WandBOutput(
            run_name=f"eval-{task_name}-{pathlib.Path(logdir_str).name}",
            config=method_config,
            project=method_config.get("wandb_project", "world-model-carla"),
        ))
        
    logger = embodied.Logger(step, outputs)

    # --- Environment Wrapping ---
    from embodied.envs import from_gym
    env = from_gym.FromGym(env)
    env = wrap_env(env, method_config)
    env = embodied.BatchEnv([env], parallel=False)

    # Instantiate Agent using the Unified WorldModelAgent
    agent = WorldModelAgent(env.obs_space, env.act_space, step, method_config)

    args = embodied.Config(
        **method_config.run,
        logdir=logdir_str,
        batch_steps=method_config.batch_size * method_config.batch_length,
    )
    eval_only(agent, env, logger, args)


if __name__ == "__main__":
    main()
