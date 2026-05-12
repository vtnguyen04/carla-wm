"""
End-to-End Sanity Check for Twister RL Pipeline
=================================================
Validates: Config loading, input shapes, module architecture,
loss computation, gradient flow, hyperparameters, and optimizer states.

Usage:
    cd /home/quynhthu/Documents/world-model/carTwister
    .venv/bin/python torch_wm/rl/tests/test_e2e_sanity.py
"""

import sys, pathlib
# Setup path like train.py does
root = str(pathlib.Path(__file__).resolve().parent.parent.parent.parent)
if root not in sys.path:
    sys.path.insert(0, root)

import torch
import numpy as np
import traceback
from collections import OrderedDict

# ── Utilities ──
PASS = "\033[92m✔ PASS\033[0m"
FAIL = "\033[91m✘ FAIL\033[0m"
WARN = "\033[93m⚠ WARN\033[0m"
results = OrderedDict()

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results[name] = condition
    print(f"  {status}  {name}" + (f"  ({detail})" if detail else ""))
    return condition

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════
# SECTION 1: Config Loading
# ══════════════════════════════════════════════════════════════
section("1. CONFIG LOADING")

import ruamel.yaml as yaml
config_path = pathlib.Path(root) / "torch_wm" / "rl" / "config" / "twister.yaml"
check("Config file exists", config_path.exists(), str(config_path))

raw_configs = yaml.YAML(typ="safe").load(config_path.read_text())
check("YAML parsed successfully", "defaults" in raw_configs)

import embodied
config = embodied.Config(raw_configs["defaults"])
check("embodied.Config created", config is not None)

# Critical hyperparameters
check("dynamics_type = tssm", config.dynamics_type == "tssm", f"got: {config.dynamics_type}")
check("stoch_size = 16", config.stoch_size == 16, f"got: {config.stoch_size}")
check("discrete = 16", config.discrete == 16, f"got: {config.discrete}")
check("hidden_size = 1024", config.hidden_size == 1024, f"got: {config.hidden_size}")
check("batch_size = 32", config.batch_size == 32, f"got: {config.batch_size}")
check("batch_length = 64", config.batch_length == 64, f"got: {config.batch_length}")
check("H (imag horizon) = 15", config.H == 15, f"got: {config.H}")
check("gamma = 0.99", config.gamma == 0.99, f"got: {config.gamma}")
check("lambda_td = 0.95", config.lambda_td == 0.95, f"got: {config.lambda_td}")
check("train_ratio = 512", config.train_ratio == 512, f"got: {config.train_ratio}")

# Optimizer LRs
check("model_opt.lr = 3e-4", config.model_opt["lr"] == 3e-4, f"got: {config.model_opt['lr']}")
check("actor_opt.lr = 1e-4", config.actor_opt["lr"] == 1e-4, f"got: {config.actor_opt['lr']}")
check("critic_opt.lr = 1e-4", config.critic_opt["lr"] == 1e-4, f"got: {config.critic_opt['lr']}")
check("model_opt.clip = 1000", config.model_opt["clip"] == 1000.0, f"got: {config.model_opt['clip']}")
check("actor_opt.clip = 100", config.actor_opt["clip"] == 100.0, f"got: {config.actor_opt['clip']}")
check("critic_opt.clip = 100", config.critic_opt["clip"] == 100.0, f"got: {config.critic_opt['clip']}")

# Action space config
action_cfg = config.env_params["action"]
discrete_acc = action_cfg["discrete_acc"]
discrete_steer = action_cfg["discrete_steer"]
n_acc = len(discrete_acc)
n_steer = len(discrete_steer)
num_actions = n_acc * n_steer
check(f"num_actions = {n_acc}*{n_steer} = {num_actions}", num_actions == 44, f"got: {num_actions}")

# Loss scales (JAX parity)
ls = config.loss_scales
check("loss_scales.image = 1.5", ls["image"] == 1.5, f"got: {ls['image']}")
check("loss_scales.dyn = 0.5", ls["dyn"] == 0.5, f"got: {ls['dyn']}")
check("loss_scales.rep = 0.1", ls["rep"] == 0.1, f"got: {ls['rep']}")
check("loss_scales.reward = 1.0", ls["reward"] == 1.0, f"got: {ls['reward']}")

# Module config
modules = config.modules
kl_cfg = modules["losses"]["kl"]
check("KL enabled", kl_cfg["enabled"] == True)
check("KL free_nats = 1.0", kl_cfg["free_nats"] == 1.0, f"got: {kl_cfg['free_nats']}")
check("KL prior_scale = 0.5", kl_cfg["prior_scale"] == 0.5, f"got: {kl_cfg['prior_scale']}")
check("KL post_scale = 0.1", kl_cfg["post_scale"] == 0.1, f"got: {kl_cfg['post_scale']}")

# Entropy scale (must be actent=1e-2 for JAX parity)
actent = config.run.get("actent", None)
check("run.actent = 1e-2 (JAX parity)", actent == 1e-2, f"got: {actent}")

# Head architecture
check("reward_head layers = 5", config.reward_head["layers"] == 5, f"got: {config.reward_head['layers']}")
check("actor layers = 5", config.actor["layers"] == 5, f"got: {config.actor['layers']}")
check("critic layers = 5", config.critic["layers"] == 5, f"got: {config.critic['layers']}")
check("cont_head layers = 5", config.cont_head["layers"] == 5, f"got: {config.cont_head['layers']}")

# ══════════════════════════════════════════════════════════════
# SECTION 2: Model Instantiation
# ══════════════════════════════════════════════════════════════
section("2. MODEL INSTANTIATION")

from torch_wm.structs import AttrDict
from torch_wm.rl.agent import WorldModelAgent

# Create mock obs/act spaces matching CARLA
class MockSpace:
    def __init__(self, shape, discrete=False, dtype=np.float32, low=0, high=1):
        self.shape = shape
        self.discrete = discrete
        self.dtype = dtype
        self.low = low
        self.high = high

obs_space = {
    "camera": MockSpace((128, 128, 3), dtype=np.uint8),
    "birdeye_wpt": MockSpace((128, 128, 3), dtype=np.uint8),
    "is_first": MockSpace((), dtype=bool),
    "is_last": MockSpace((), dtype=bool),
    "is_terminal": MockSpace((), dtype=bool),
    "reward": MockSpace((), dtype=np.float32),
}
act_space = {
    "action": MockSpace((num_actions,), discrete=True),
    "reset": MockSpace((), discrete=True),
}

step = embodied.Counter()
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    agent = WorldModelAgent(obs_space, act_space, step, config)
    check("WorldModelAgent created", True)
except Exception as e:
    check("WorldModelAgent created", False, str(e))
    traceback.print_exc()
    sys.exit(1)

model = agent.model
check("UnifiedAgent on device", str(next(model.parameters()).device).startswith(device[:3]),
      f"got: {next(model.parameters()).device}")
check("model.compiled = True", model.compiled)
check(f"num_actions = {model.dynamics_model.num_actions}", model.dynamics_model.num_actions == num_actions,
      f"got: {model.dynamics_model.num_actions}")

# ══════════════════════════════════════════════════════════════
# SECTION 3: Module Architecture Verification
# ══════════════════════════════════════════════════════════════
section("3. MODULE ARCHITECTURE")

check("encoder_network exists", hasattr(model, "encoder_network"))
check("decoder_network exists", hasattr(model, "decoder_network"))
check("dynamics_model exists", hasattr(model, "dynamics_model"))
check("reward_network exists", hasattr(model, "reward_network"))
check("continue_network exists", hasattr(model, "continue_network"))
check("policy_network exists", hasattr(model, "policy_network"))
check("value_network exists", hasattr(model, "value_network"))
check("v_target exists", hasattr(model, "v_target"))
check("contrastive_network exists", hasattr(model, "contrastive_network"))
check("world_model exists", hasattr(model, "world_model"))
check("actor_model exists", hasattr(model, "actor_model"))
check("critic_model exists", hasattr(model, "critic_model"))

# Check optimizer isolation
wm_opt = model.world_model.optimizer
actor_opt = model.actor_model.optimizer
critic_opt = model.critic_model.optimizer
check("WM optimizer exists", wm_opt is not None)
check("Actor optimizer exists", actor_opt is not None)
check("Critic optimizer exists", critic_opt is not None)

wm_param_ids = set(id(p) for g in wm_opt.param_groups for p in g["params"])
actor_param_ids = set(id(p) for g in actor_opt.param_groups for p in g["params"])
critic_param_ids = set(id(p) for g in critic_opt.param_groups for p in g["params"])

check("WM/Actor params disjoint", len(wm_param_ids & actor_param_ids) == 0,
      f"overlap: {len(wm_param_ids & actor_param_ids)}")
check("WM/Critic params disjoint", len(wm_param_ids & critic_param_ids) == 0,
      f"overlap: {len(wm_param_ids & critic_param_ids)}")
check("Actor/Critic params disjoint", len(actor_param_ids & critic_param_ids) == 0,
      f"overlap: {len(actor_param_ids & critic_param_ids)}")

# Check policy network params are in actor optimizer
policy_param_ids = set(id(p) for p in model.policy_network.parameters())
check("Policy params in Actor opt", policy_param_ids.issubset(actor_param_ids),
      f"policy: {len(policy_param_ids)}, in actor opt: {len(policy_param_ids & actor_param_ids)}")

# Check value network params are in critic optimizer
value_param_ids = set(id(p) for p in model.value_network.parameters())
check("Value params in Critic opt", value_param_ids.issubset(critic_param_ids),
      f"value: {len(value_param_ids)}, in critic opt: {len(value_param_ids & critic_param_ids)}")

# Check v_target has no grad
v_target_grads = [p.requires_grad for p in model.v_target.parameters()]
check("v_target params frozen", not any(v_target_grads),
      f"requires_grad count: {sum(v_target_grads)}/{len(v_target_grads)}")

# Learning rates
for name, opt, expected_lr in [
    ("WM", wm_opt, 3e-4), ("Actor", actor_opt, 1e-4), ("Critic", critic_opt, 1e-4)
]:
    actual_lr = opt.param_groups[0]["lr"]
    if isinstance(actual_lr, (int, float)):
        ok = abs(float(actual_lr) - expected_lr) < 1e-6
        detail = f"got: {actual_lr}"
    else:
        base = getattr(actual_lr, 'max_lr', getattr(actual_lr, 'base_lr', None))
        if base is not None:
            ok = abs(float(base) - expected_lr) < 1e-6
            detail = f"scheduler, base_lr: {base}"
        else:
            ok = True
            detail = f"scheduler-wrapped ({type(actual_lr).__name__})"
    check(f"{name} LR ~ {expected_lr}", ok, detail)

# Loss manager active losses
active_losses = [l.name() for l in model.world_model.loss_manager.active_losses]
check("KL loss active", "kl" in active_losses, f"active: {active_losses}")
check("Reconstruction loss active", "reconstruction" in active_losses, f"active: {active_losses}")
check("Reward loss active", "reward" in active_losses, f"active: {active_losses}")
check("Discount loss active", "discount" in active_losses, f"active: {active_losses}")

# ══════════════════════════════════════════════════════════════
# SECTION 4: Input Pipeline Test (Fake Batch)
# ══════════════════════════════════════════════════════════════
section("4. INPUT PIPELINE")

# Use sequence length >= att_context_left (32) + margin for TSSM
B = 2   # Small batch for speed
L = 64  # Must match batch_length for TSSM context window

# Create fake batch (as would come from replay buffer, numpy, HWC format)
fake_batch = {
    "camera": np.random.randint(0, 255, (B, L, 128, 128, 3), dtype=np.uint8),
    "birdeye_wpt": np.random.randint(0, 255, (B, L, 128, 128, 3), dtype=np.uint8),
    "action": np.eye(num_actions, dtype=np.float32)[np.random.randint(0, num_actions, (B, L))],
    "reward": np.random.randn(B, L).astype(np.float32),
    "is_first": np.zeros((B, L), dtype=bool),
    "is_last": np.zeros((B, L), dtype=bool),
    "is_terminal": np.zeros((B, L), dtype=bool),
}
fake_batch["is_first"][:, 0] = True

# Test preprocessing
from torch_wm.utils.preprocessing import preprocess_obs_for_agent
processed = preprocess_obs_for_agent(fake_batch, device)

check("camera preprocessed shape", processed["camera"].shape == (B, L, 3, 128, 128),
      f"got: {processed['camera'].shape}")
check("birdeye_wpt preprocessed shape", processed["birdeye_wpt"].shape == (B, L, 3, 128, 128),
      f"got: {processed['birdeye_wpt'].shape}")
check("camera dtype float", processed["camera"].dtype in (torch.float32, torch.float16),
      f"got: {processed['camera'].dtype}")
check("camera value range [-0.5, 0.5]",
      processed["camera"].min() >= -0.5 and processed["camera"].max() <= 0.5,
      f"range: [{processed['camera'].min():.3f}, {processed['camera'].max():.3f}]")
check("action shape", processed["action"].shape == (B, L, num_actions),
      f"got: {processed['action'].shape}")
check("action is one-hot", processed["action"].sum(-1).allclose(torch.ones(B, L, device=device)),
      f"sum per step: {processed['action'].sum(-1).mean():.3f}")
check("reward shape", processed["reward"].shape == (B, L),
      f"got: {processed['reward'].shape}")

# ══════════════════════════════════════════════════════════════
# SECTION 5–7: Full Train Step (WM → Actor → Critic in sequence)
# ══════════════════════════════════════════════════════════════
# NOTE: Actor and Critic forward() depend on `model.detached_posts/feats/returns`
# being populated by a prior WM forward pass. We must call them in order.
# The cleanest way is agent.train() which orchestrates this correctly.

section("5-7. FULL TRAIN STEP (agent.train)")

model.train()
try:
    outs, state, metrics = agent.train(fake_batch, state=None)
    check("agent.train() succeeds", True)
    check("Returns 3-tuple", isinstance(outs, dict) and metrics is not None)

    # Check critical metrics exist
    metric_keys = list(metrics.keys())
    check("wm_loss in metrics", "wm_loss" in metrics or "wm/loss" in metrics,
          f"keys: {metric_keys[:15]}")

    # Check all metric values are finite
    nan_metrics = []
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            if not np.isfinite(v): nan_metrics.append(k)
        elif isinstance(v, torch.Tensor) and v.dim() == 0:
            if not torch.isfinite(v): nan_metrics.append(k)
    check("All metrics finite", len(nan_metrics) == 0, f"NaN/Inf: {nan_metrics}")

    # Check sub-losses exist per component
    has_actor_loss = any("actor" in k for k in metric_keys)
    has_critic_loss = any("critic" in k for k in metric_keys)
    has_wm_loss = any("wm" in k for k in metric_keys)
    check("WM loss metrics present", has_wm_loss, f"wm keys: {[k for k in metric_keys if 'wm' in k][:5]}")
    check("Actor loss metrics present", has_actor_loss, f"actor keys: {[k for k in metric_keys if 'actor' in k][:5]}")
    check("Critic loss metrics present", has_critic_loss, f"critic keys: {[k for k in metric_keys if 'critic' in k][:5]}")

    # Print all metrics for visibility
    print(f"\n  📊 Metrics ({len(metrics)} keys):")
    for k, v in sorted(metrics.items()):
        if isinstance(v, torch.Tensor) and v.dim() == 0:
            print(f"     {k}: {v.item():.6f}")
        elif isinstance(v, (int, float)):
            print(f"     {k}: {v:.6f}")

except Exception as e:
    check("agent.train() succeeds", False, str(e)[:200])
    traceback.print_exc()

# ══════════════════════════════════════════════════════════════
# SECTION 8: Gradient Flow Verification
# ══════════════════════════════════════════════════════════════
section("8. GRADIENT FLOW")

# After agent.train(), the optimizers have already stepped and zeroed grads.
# To verify gradient flow, we manually do a forward-backward WITHOUT stepping.

def check_grad_flow(name, param_iter):
    params = list(param_iter)
    has_grad = sum(1 for p in params if p.grad is not None and p.grad.abs().sum() > 0)
    total = len(params)
    check(f"{name} grads populated", has_grad > 0, f"{has_grad}/{total} params have nonzero grad")
    nan_count = sum(1 for p in params if p.grad is not None and not torch.isfinite(p.grad).all())
    check(f"{name} grads finite (no NaN/Inf)", nan_count == 0, f"{nan_count} params have NaN/Inf grad")

# --- WM gradient flow ---
print("\n  [WM gradient test]")
# Prepare inputs
s = {k: v for k, v in processed.items()
     if k not in ("action", "reward", "is_first", "is_last", "is_terminal")}
a = processed["action"]
r = processed["reward"]
d_cont = 1.0 - processed.get("is_terminal", torch.zeros_like(r)).float()
f_flag = processed["is_first"].float()
if r.dim() == 2: r = r.unsqueeze(-1)
if d_cont.dim() == 2: d_cont = d_cont.unsqueeze(-1)
if f_flag.dim() == 2: f_flag = f_flag.unsqueeze(-1)
inputs_tuple = (s, a, r, d_cont, f_flag)

model.train()
wm_opt.zero_grad()
# Use train_step which does backward + step internally
# But we need to check grads BEFORE they're zeroed.
# The simplest approach: call forward_model + backward manually.
try:
    wm_loss_dict, wm_mets, _, _ = model.world_model.forward_model(inputs_tuple, inputs_tuple)
    total_wm_loss = sum(v for v in wm_loss_dict.values() if isinstance(v, torch.Tensor))
    total_wm_loss.backward()
    
    check_grad_flow("Encoder", model.encoder_network.parameters())
    check_grad_flow("Dynamics", model.dynamics_model.parameters())
    check_grad_flow("Reward head", model.reward_network.parameters())
    check_grad_flow("Decoder", model.decoder_network.parameters())
    
    wm_opt.zero_grad()  # Clean up
except Exception as e:
    check("WM gradient test", False, str(e)[:100])
    traceback.print_exc()

# --- Actor gradient flow ---
print("\n  [Actor gradient test]")
try:
    # Actor needs WM forward to have populated detached_posts first
    # Re-run WM forward without backward to populate caches
    with torch.no_grad():
        model.world_model.forward_model(inputs_tuple, inputs_tuple)
    
    actor_opt.zero_grad()
    act_loss_dict, _, _, _ = model.actor_model.forward_model(inputs_tuple, inputs_tuple)
    total_act_loss = sum(v for v in act_loss_dict.values() if isinstance(v, torch.Tensor))
    total_act_loss.backward()
    
    check_grad_flow("Policy network", model.policy_network.parameters())
    
    # Verify encoder grads are NOT populated (isolation: actor should not backprop into encoder)
    enc_grads = sum(1 for p in model.encoder_network.parameters() 
                    if p.grad is not None and p.grad.abs().sum() > 0)
    check("Encoder isolated from Actor", enc_grads == 0,
          f"{enc_grads} encoder params have grad from actor backward")
    
    actor_opt.zero_grad()
except Exception as e:
    check("Actor gradient test", False, str(e)[:100])
    traceback.print_exc()

# --- Critic gradient flow ---
print("\n  [Critic gradient test]")
try:
    critic_opt.zero_grad()
    crit_loss_dict, _, _, _ = model.critic_model.forward_model(inputs_tuple, inputs_tuple)
    total_crit_loss = sum(v for v in crit_loss_dict.values() if isinstance(v, torch.Tensor))
    total_crit_loss.backward()
    
    check_grad_flow("Value network", model.value_network.parameters())
    
    # Verify policy grads are NOT populated (isolation)
    pol_grads = sum(1 for p in model.policy_network.parameters() 
                    if p.grad is not None and p.grad.abs().sum() > 0)
    check("Policy isolated from Critic", pol_grads == 0,
          f"{pol_grads} policy params have grad from critic backward")
    
    critic_opt.zero_grad()
except Exception as e:
    check("Critic gradient test", False, str(e)[:100])
    traceback.print_exc()

# ══════════════════════════════════════════════════════════════
# SECTION 9: Policy Output Validation
# ══════════════════════════════════════════════════════════════
section("9. POLICY OUTPUT")

model.eval()

single_obs = {
    "camera": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
    "birdeye_wpt": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
    "is_first": np.array(True),
    "reward": np.float32(0.0),
}

try:
    out, state = agent.policy(single_obs, state=None, mode="train")
    action = out["action"]
    check("Policy returns action", "action" in out)
    check("Action is numpy", isinstance(action, np.ndarray), f"type: {type(action)}")
    check(f"Action shape = (1, {num_actions})", action.shape == (1, num_actions),
          f"got: {action.shape}")
    check("Action is one-hot-like", np.allclose(action.sum(-1), 1.0, atol=0.01),
          f"sum: {action.sum(-1)}")
    check("Action values in [0,1]", action.min() >= -0.01 and action.max() <= 1.01,
          f"range: [{action.min():.3f}, {action.max():.3f}]")

    check("State returned", state is not None)
    if state:
        for k, v in state.items():
            if k == "hidden": continue
            if v is not None:
                check(f"  state['{k}'] is numpy", isinstance(v, np.ndarray), f"type: {type(v)}")
except Exception as e:
    check("Policy forward succeeds", False, str(e)[:100])
    traceback.print_exc()

# ══════════════════════════════════════════════════════════════
# SECTION 10: Steering Smoothing Guard
# ══════════════════════════════════════════════════════════════
section("10. STEERING SMOOTHING GUARD")

import inspect
from carla_env.carla_base_env import CarlaBaseEnv

src = inspect.getsource(CarlaBaseEnv.get_vehicle_control)
check("Smoothing guarded by eval check",
      "self._config.eval" in src or "self._config, 'eval'" in src,
      "Found eval guard in get_vehicle_control")
check("No unconditional smoothing",
      "alpha_steer" not in src.split("self._config.eval")[0].split("def get_vehicle_control")[1]
      if "self._config.eval" in src else False,
      "alpha_steer only appears after eval check")


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
section("SUMMARY")

total = len(results)
passed = sum(1 for v in results.values() if v)
failed = total - passed

print(f"\n  Total:  {total}")
print(f"  Passed: {passed}")
print(f"  Failed: {failed}")

if failed > 0:
    print(f"\n  ❌ FAILED CHECKS:")
    for name, ok in results.items():
        if not ok:
            print(f"     • {name}")

print()
sys.exit(0 if failed == 0 else 1)
