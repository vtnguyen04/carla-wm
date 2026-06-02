import datetime
import warnings
import embodied
import pickle
import ruamel.yaml as yaml
from tqdm import tqdm
import numpy as np
import sys

import carla_env
from torch_wm.rl.agent import WorldModelAgent
from carla_env.toolkit.utils import get_logger

log = get_logger(log_dir=".", job_name="collect")

warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")


def wrap_env(env, config):
    args = config.wrapper
    env = embodied.wrappers.InfoWrapper(env)
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


def collect(agent, env, replay, logger, args):
    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    log.info(f"Logdir: {logdir}")
    should_expl = embodied.when.Until(args.expl_until)
    step = logger.step
    metrics = embodied.Metrics()
    log.info(f"Observation space:\n{embodied.format(env.obs_space)}")
    log.info(f"Action space:\n{embodied.format(env.act_space)}")

    timer = embodied.Timer()
    timer.wrap("agent", agent, ["policy"])
    timer.wrap("env", env, ["step"])
    timer.wrap("replay", replay, ["add", "save"])
    timer.wrap("logger", logger, ["write"])

    def per_episode(ep):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        logger.add({"length": length, "score": score}, prefix="episode")
        log.info(f"Episode has {length} steps and return {score:.1f}.")
        metrics.add({"score": score}, prefix="episode")

    driver = embodied.Driver(env)
    driver.on_episode(lambda ep, ep_info, worker: per_episode(ep))
    driver.on_step(lambda _, __, ___: step.increment())
    driver.on_step(lambda tran, _, worker: replay.add(tran, worker))

    checkpoint = embodied.Checkpoint(logdir / "checkpoint.ckpt")
    checkpoint.agent = agent
    if args.from_checkpoint and args.from_checkpoint != 'none':
        try:
            checkpoint.load(args.from_checkpoint, keys=['agent'])
            log.info(f"Loaded agent checkpoint from {args.from_checkpoint}")
        except Exception as e:
            log.error(f"Could not load checkpoint from {args.from_checkpoint}: {e}")
            log.info("Continuing with random policy.")
    else:
        log.info("No checkpoint provided, using random policy for collection.")

    log.info("Start collection loop.")
    if args.from_checkpoint and args.from_checkpoint != 'none':
        policy = lambda *args: agent.policy(*args, mode="explore" if should_expl(step) else "eval")
    else:
        policy = embodied.RandomAgent(env.act_space, args.actor_dist_disc).policy

    # Load replay buffer if it exists
    try:
        replay.load(logdir / "replay", continue_obs=False)
        log.info(f"Loaded existing replay buffer with {len(replay)} steps.")
    except Exception as e:
        log.info(f"Could not load replay buffer, starting new one: {e}")


    with tqdm(total=args.steps, initial=int(step.value)) as pbar:
        while step < args.steps:
            driver(policy, steps=100)
            pbar.update(100)

    log.info(f"Collection finished. Total steps: {step.value}. Saving replay buffer...")
    replay.save()
    log.info("Replay buffer saved.")
    logger.write()


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if "--help" in argv:
        print("Usage: collect.py")
        model_configs = yaml.YAML(typ="safe").load(
            (embodied.Path(__file__).parent / "config" / "dreamerv3.yaml").read()
        )
        config = embodied.Config(model_configs["defaults"])
        config = config.update({"from_checkpoint": "none"})
        # We don't load env config here to avoid CARLA connection
        embodied.Flags(config).parse(argv)

    model_configs = yaml.YAML(typ="safe").load(
        (embodied.Path(__file__).parent / "config" / "dreamerv3.yaml").read()
    )
    temp_flags = embodied.Flags(method="twister", model_size="defaults")
    temp_parsed, temp_other = temp_flags.parse_known(argv)
    method = temp_parsed.method
    model_size = temp_parsed.model_size

    if model_size not in model_configs:
        log.warning(f"Unknown model_size: {model_size}. Falling back to 'defaults'")
        model_size = "defaults"

    config = embodied.Config(model_configs["defaults"])
    if model_size != "defaults":
        config = config.update(model_configs[model_size])

    parsed, other = embodied.Flags(task=["carla_navigation"]).parse_known(temp_other)
    for name in parsed.task:
        log.info(f"Using task: {name}")
        env, env_config = carla_env.create_task(name, other)
        config = config.update(env_config)

    # Add a flag for from_checkpoint
    config = config.update({"from_checkpoint": "none"})
    config = embodied.Flags(config).parse(other)
    log.info(config)

    logdir = embodied.Path(config.logdir)
    step = embodied.Counter()
    logger = embodied.Logger(
        step,
        [
            embodied.logger.TerminalOutput(pattern=r".*reward.*|.*return.*|.*loss.*"),
            embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
            embodied.logger.TensorBoardOutput(logdir),
        ],
    )

    from embodied.envs import from_gym

    wm_config = config
    env = from_gym.FromGym(env)
    env = wrap_env(env, wm_config)
    env = embodied.BatchEnv([env], parallel=False)

    log.info("Saving observation and action spaces.")
    spaces = {'obs_space': env.obs_space, 'act_space': env.act_space}
    with open(logdir / 'spaces.pkl', 'wb') as f:
        pickle.dump(spaces, f)
    log.info(f"Spaces saved to {logdir / 'spaces.pkl'}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config_filename = f"config_{timestamp}.yaml"
    config.save(str(logdir / config_filename))
    log.info(f"[Collect] Config saved to {logdir / config_filename}")

    agent = WorldModelAgent(env.obs_space, env.act_space, step, wm_config)
    replay = embodied.replay.Uniform(
        wm_config.batch_length, wm_config.replay_size, logdir / "replay"
    )

    args_kwargs = dict(wm_config.run)
    args_kwargs.update({
        "logdir": wm_config.logdir,
        "steps": wm_config.get('collect_steps', wm_config.get('steps', 1e6)),
        "from_checkpoint": wm_config.get('from_checkpoint', 'none'),
    })
    args = embodied.Config(**args_kwargs)
    collect(agent, env, replay, logger, args)


if __name__ == "__main__":
    main()
