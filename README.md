# CARLA World Model

Transformer and recurrent world-model training for autonomous driving experiments in CARLA.

This repository combines a CARLA driving environment, expert data collection scripts, and PyTorch world-model agents based on DreamerV3-style RSSM, TWISTER-style TSSM, V-JEPA predictors, and diffusion or SiT policy components.

## Table of Contents

- [About](#about)
- [Features](#features)
- [Repository Layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Testing](#testing)
- [Development Workflow](#development-workflow)
- [Troubleshooting](#troubleshooting)
- [Project Status](#project-status)
- [Contributing](#contributing)
- [License](#license)

## About

CARLA World Model is a research codebase for learning compact latent dynamics from driving observations and using those dynamics for reinforcement learning. The main training path wraps a CARLA Gym-style task with the `embodied` runtime, stores online experience in replay, and trains a modular world-model agent.

The current code supports three model families through YAML configuration:

- `dreamerv3`: RSSM dynamics with DreamerV3-style actor, critic, decoder, reward, and continue heads.
- `twister`: TSSM dynamics with transformer context over latent states.
- `vjepa`: V-JEPA encoder or predictor experiments for visual representation learning.

It also includes experimental diffusion and SiT policy networks for stochastic action generation and policy training research.

## Features

- CARLA navigation task integration through `carla_env`.
- Online RL training with replay buffers, TensorBoard, JSONL metrics, and optional W&B logging.
- Expert policy data collection with modular steering and acceleration discretization.
- Multi-modal encoders and decoders for camera and bird-eye waypoint observations.
- RSSM and TSSM dynamics modules.
- Dense reward, value, and continue heads.
- Diffusion and SiT policy network experiments.
- V-JEPA encoder and predictor integration tests.
- Pytest suite with coverage gate.
- GitHub Actions CI for compile checks and tests.

## Repository Layout

```text
.
├── carla_env/                 # CARLA task, configs, planners, sensors, managers
├── embodied/                  # Embodied runtime wrappers and support code
├── scripts/                   # Training, evaluation, deployment, and data collection scripts
├── tests/                     # Unit, integration, RL, V-JEPA, and module tests
├── torch_wm/                  # World-model agents, dynamics, networks, losses, RL loop
│   ├── models/                # WMAgent, actor, critic, world model, custom actor models
│   ├── modules/               # Blocks, dynamics, networks, losses, planners, controllers
│   ├── rl/                    # RL agent wrapper, train/eval code, configs, loggers
│   └── utils/                 # Collate, checkpointing, tensors, preprocessing, YAML helpers
├── pyproject.toml             # Python metadata, dependencies, pytest and coverage config
└── uv.lock                    # Locked dependency graph
```

## Requirements

Core requirements:

- Python `3.10.x`
- `uv`
- PyTorch and torchvision
- CARLA `0.9.16`
- A CARLA simulator installation for environment interaction
- CUDA-capable GPU recommended for training

Optional but commonly used:

- TensorBoard
- Weights & Biases
- `nc` and `fuser` for the helper shell scripts that monitor CARLA ports

The training scripts expect `CARLA_ROOT` to point to the directory that contains `CarlaUE4.sh`:

```bash
export CARLA_ROOT=/path/to/CARLA_0.9.16
```

## Installation

Clone the repository:

```bash
git clone https://github.com/vtnguyen04/carla-wm.git
cd carla-wm
```

Install dependencies from the lockfile:

```bash
uv sync --locked
```

If you need to run Python directly instead of `uv run`, activate the environment:

```bash
source .venv/bin/activate
```

Set CARLA-related environment variables:

```bash
export CARLA_ROOT=/path/to/CARLA_0.9.16
export PYTHONPATH="$PWD:${PYTHONPATH}"
```

Verify the Python package imports and tests collect:

```bash
uv run python -m compileall -q carla_env embodied scripts tests torch_wm
uv run pytest -q --no-cov
```

## Configuration

Model and training defaults live in:

```text
torch_wm/rl/config/dreamerv3.yaml
torch_wm/rl/config/twister.yaml
torch_wm/rl/config/vjepa.yaml
```

Important config fields:

| Field | Description |
| --- | --- |
| `dynamics_type` | Dynamics backend. Examples: `rssm`, `tssm`, `vjepa_predictor`. |
| `policy_type` | Policy backend. Examples: `default_policy`, `diffusion`, `sit`. |
| `actor_model_type` | Actor training module. Examples: `default_actor`, `diffusion_actor`. |
| `num_actions` | Size of the discrete one-hot action space. |
| `batch_size`, `batch_length` | Replay batch dimensions. |
| `train_ratio`, `train_fill`, `replay_min` | Online RL replay and update schedule. |
| `logdir` | Output directory for checkpoints, metrics, and replay. |
| `wandb`, `tensorboard` | Enable or disable external logging outputs. |
| `env_params` | CARLA task, observation, action, display, and world parameters. |

CARLA environment defaults live under:

```text
carla_env/configs/
```

Most runtime overrides can be passed as flags. For example:

```bash
uv run python -m torch_wm.rl.train \
  --method twister \
  --env.world.carla_port 2000 \
  --device cuda \
  --seed 0
```

## Usage

### Run Online Training

Start a CARLA server separately, then run:

```bash
uv run python -m torch_wm.rl.train --method twister --env.world.carla_port 2000
```

Use DreamerV3-style RSSM instead:

```bash
uv run python -m torch_wm.rl.train --method dreamerv3 --env.world.carla_port 2000
```

### Run Training with the CARLA Watchdog Script

The helper script starts CARLA if the selected port is unavailable and restarts training if it exits:

```bash
bash scripts/train_dm3.sh 2000 0 --method twister
```

Arguments:

- `2000`: CARLA RPC port
- `0`: GPU id for `CUDA_VISIBLE_DEVICES`
- Remaining args: forwarded to `torch_wm.rl.train`

### Evaluate a Checkpoint

```bash
bash scripts/eval_dm3.sh 2000 0 runs/wm_twister/checkpoint.ckpt --method twister
```

Arguments:

- `2000`: CARLA RPC port
- `0`: GPU id
- `runs/wm_twister/checkpoint.ckpt`: checkpoint path
- Remaining args: forwarded to evaluation

### Collect Expert Data

Run the expert data collector:

```bash
uv run python scripts/collect_expert_data.py
```

The collector creates a CARLA navigation environment, applies a modular expert controller, and records trajectories for offline inspection or training workflows.

### Run Shape and Gradient Sanity Checks

```bash
uv run python scripts/test_shapes_gradients.py
uv run python scripts/training/sanity_check.py
```

## Testing

Run the full test suite with the configured coverage gate:

```bash
uv run pytest
```

Current pytest configuration:

- Test root: `tests`
- Coverage target: `torch_wm`
- Coverage threshold: `80%`
- Coverage report: terminal missing-lines report

Run a fast compile check:

```bash
uv run python -m compileall -q carla_env embodied scripts tests torch_wm
```

Run targeted tests:

```bash
uv run pytest -q tests/test_modules/test_diffusion_policy.py --no-cov
uv run pytest -q tests/test_multi_algo.py --no-cov
uv run pytest -q tests/vjepa_wm --no-cov
```

## Development Workflow

Recommended workflow:

1. Create a feature branch from `dev`.
2. Keep commits scoped by task.
3. Run compile checks and targeted tests locally.
4. Run the full suite before opening or updating a pull request.
5. Merge feature branches into `dev`.
6. Open a PR from `dev` to `main`.

Example:

```bash
git switch dev
git pull
git switch -c feature/my-change

uv run python -m compileall -q carla_env embodied scripts tests torch_wm
uv run pytest
```

Commit message style used in this repository:

```text
feat(scope): add new behavior
fix(scope): correct broken behavior
refactor(scope): restructure without changing intended behavior
test(scope): add or update tests
chore(scope): update maintenance files
```

## Troubleshooting

### CARLA Does Not Start

Check that `CARLA_ROOT` is set and points to a valid CARLA installation:

```bash
echo "$CARLA_ROOT"
ls "$CARLA_ROOT/CarlaUE4.sh"
```

Check that the selected port is free:

```bash
nc -z localhost 2000
```

### Python Cannot Import CARLA

Confirm that CARLA's Python API is available to the active Python environment. The project pins `carla==0.9.16`, but local CARLA simulator and Python API paths still need to be compatible.

### CUDA Out of Memory

Useful mitigations:

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
```

Then reduce one or more of:

- `batch_size`
- `batch_length`
- `train_ratio`
- `hidden_size`
- `num_blocks_trans`
- `att_context_left`

### W&B Starts Unexpectedly

Set `wandb: False` in the selected YAML config or override it in the runtime config.

### Tests Pass Locally but CI Fails

Check for:

- Absolute filesystem paths in tests
- CARLA-dependent tests that should be mocked or skipped
- Generated debug files accidentally added to the repository
- Differences between local CUDA availability and CPU-only CI

## Project Status

This is an active research and engineering codebase. The tested surface includes core model modules, losses, dynamics, RL wrappers, V-JEPA integration, diffusion or SiT policy paths, and training setup logic. CARLA runtime behavior still depends on simulator availability, local GPU resources, and environment configuration.

## Contributing

Pull requests are welcome. For substantial changes, open an issue or draft PR first so the architecture and test plan can be discussed.

Before submitting:

```bash
uv run python -m compileall -q carla_env embodied scripts tests torch_wm
uv run pytest
```

Please keep unrelated debug files, scratch outputs, replay buffers, and logs out of commits.

## License

No root-level license file is currently included in this repository. Add a `LICENSE` file before distributing the project as open source or reusing it outside its current private/research context.

