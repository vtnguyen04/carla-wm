import torch
import numpy as np
import embodied
from torch_wm.models.unified_agent import UnifiedAgent
from torch_wm.structs import AttrDict

class WorldModelAgent(embodied.Agent):
    def __init__(self, obs_space, act_space, step, config):
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space
        self.step = step
        
        # Mapping config from embodied to World Model
        wm_config = self._prepare_wm_config(config)
        
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        
        # Create PyTorch-compatible obs space (CHW) for WMAgent model initialization
        self.pt_obs_space = {}
        for k, v in self.obs_space.items():
            raw_shape = v.shape
            # If sequence replay, strip sequence and batch length
            # A raw CARLA camera tensor will have 3 dims (H, W, C)
            # A batched replay chunk might have 5 dims (B, T, H, W, C)
            
            # Remove sequence/batch dims for Images
            if len(raw_shape) >= 3 and raw_shape[-1] == 3:
                raw_shape = raw_shape[-3:]
            # Remove sequence/batch dims for Vectors
            elif len(raw_shape) >= 1 and k != 'camera':
                if raw_shape[-1] > 3: # If vector length > 3
                    raw_shape = raw_shape[-1:]
                
            # If shape ends in 3, it's (H, W, C) -> Permute to (C, H, W)
            if len(raw_shape) == 3 and raw_shape[-1] == 3:
                pt_shape = (raw_shape[2], raw_shape[0], raw_shape[1])
            else:
                pt_shape = raw_shape
                
            class SpaceStub:
                def __init__(self, shape): self.shape = shape
            self.pt_obs_space[k] = SpaceStub(pt_shape)
        
        # Instantiate UnifiedAgent instead of WMAgent
        self.model = UnifiedAgent(
            env_name=config.task, 
            override_config=wm_config, 
            skip_env=True,
            obs_space=self.pt_obs_space
        ).to(self.device)
        
        # Setup Exploration Module if enabled
        self.expl = None
        if config.get("expl_reward", False):
            from torch_wm.rl.expl import Disag
            self.expl = Disag(wm_config).to(self.device)

        # Pass action configuration for dynamic decoding in report()
        # They are usually in config.env_params.action or similar nested path
        action_cfg = config.get("env_params", {}).get("action", {})
        self.model.config.discrete_acc = action_cfg.get("discrete_acc", [-3.0, 0.0, 3.0])
        self.model.config.discrete_steer = action_cfg.get("discrete_steer", [-0.6, 0.0, 0.6])

        # Critical: Compile to setup optimizers and target networks
        self.model.compile()
        
        self._updates = 0

    @property
    def strategy(self):
        return self.model.strategy_name

    def _prepare_wm_config(self, config):
        c = AttrDict()
        
        # Base architecture default fallbacks (safeguards against missing YAML keys)
        defaults = {
            "dynamics_type": "rssm",
            "stoch_size": 32, "discrete": 32, "hidden_size": 256, 
            "num_blocks_trans": 1, "att_context_left": 32,
            "precision": "float32", "model_lr": 1e-4, "actor_lr": 8e-5, "critic_lr": 8e-5,
            "grad_clip": 100.0, "gamma": 0.997, "lambda_td": 0.95,
            "H": 10, "actor_entropy": 1e-3,
            "contrastive_steps": 1, "uniform_mix": 0.01,
            "contrastive_offsets": None,
            "target_value_reg": False,
            
            # Unified Strategy Flags
            "agent_strategy": "actor_critic",
            "use_ema": False,
            "ema_decay": 0.98,
            
            # Additional architectural hypers mapped from yaml purely decoupled (SOLID OCP)
            "adam_eps": 1e-5, "weight_decay": 1e-6,
            "warmup_steps": 1000, "total_steps": 1000000, "min_lr_ratio": 0.1,
            "dim_cnn": 32,
            "encoder_dim_layers": [32, 64, 128, 256],
            "decoder_dim_layers": [256, 128, 64, 32, 3]
        }
        
        c.update(defaults)
        
        # Inject all keys that exist in yaml config into model configs dynamically
        # Expanded to include common keys not in defaults
        keys_to_check = set(list(defaults.keys()) + ["dynamics_type", "discrete_steer", "discrete_acc"])
        for k in keys_to_check:
            if k in config: c[k] = config[k]
        
        # Explicitly copy 'modules' to preserve loss manager settings
        if "modules" in config:
            c.modules = config["modules"]
        
        # Calculate num_actions correctly
        if "action" in self.act_space:
            space = self.act_space["action"]
            if hasattr(space, "n"): 
                c.num_actions = space.n
                c.discrete_actions = True
            elif hasattr(space, "shape") and len(space.shape) > 0:
                # Use LAST dim as action size (shape may include batch/seq dims from replay)
                c.num_actions = space.shape[-1]
                c.discrete_actions = True  # One-hot action from CARLA discrete env
            else:
                c.num_actions = 63
                c.discrete_actions = True
        else:
            c.num_actions = 63
            c.discrete_actions = True
        
        c.env_params = AttrDict()
        env_cfg = config.get("env", {})
        if "observation" in env_cfg:
            c.env_params.observation = AttrDict(env_cfg["observation"])
        else:
            # Build observation config from obs_space
            # Shapes from replay include the sequence dim (L, H, W, C).
            # MultiEncoderNetwork expects raw sensor shape (H, W, C), so strip leading dims.
            image_keys = []
            obs_entries = {}
            enabled_keys = []
            for k, v in self.obs_space.items():
                if k in ("reward", "is_first", "is_last", "is_terminal"):
                    continue
                enabled_keys.append(k)
                # Strip all leading dims to get the raw sensor shape (last 3 dims for images)
                raw_shape = v.shape
                if len(raw_shape) >= 4 and raw_shape[-1] == 3:
                    raw_shape = raw_shape[-3:]  # (H, W, C) for images
                elif len(raw_shape) >= 2:
                    raw_shape = raw_shape[-1:]  # (D,) for vectors
                obs_entries[k] = AttrDict({
                    "shape": list(raw_shape),
                    # BEV is navigation signal only — encode but don't feed TSSM or reconstruct
                    "decode": k != "birdeye_wpt",
                    "tssm": k != "birdeye_wpt",
                })
            
            c.env_params.observation = AttrDict({
                "enabled": enabled_keys,
                **obs_entries
            })
            
        c.env_params.action = AttrDict({"discrete": c.discrete_actions, "n_cmds": c.num_actions})
        c.batch_size = config.get("batch_size", 4)
        c.batch_length = config.get("batch_length", 32)
        return c

    def _preprocess_obs(self, obs):
        from torch_wm.utils.preprocessing import preprocess_obs_for_agent
        return preprocess_obs_for_agent(obs, self.device)

    def policy(self, obs, state=None, mode="train"):
        """Select action given observation. Called every env step during online training."""
        self.model.eval()
        obs = self._preprocess_obs(obs)

        # Extract batch size and ensure batch dimension for all sensors
        first_key = next(k for k in obs if isinstance(obs[k], torch.Tensor))
        first_val = obs[first_key]
        
        # Determine if we need to add a batch dimension
        needs_unsquash = False
        if first_val.dim() == 3 and (first_key == 'camera' or first_key == 'birdeye_wpt'):
            needs_unsquash = True
        elif first_val.dim() == 1 and first_key != 'camera' and first_key != 'birdeye_wpt':
            needs_unsquash = True
            
        if needs_unsquash:
            obs = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in obs.items()}
            B = 1
        else:
            B = obs[first_key].shape[0]

        is_first = obs.get("is_first")
        if is_first is not None:
            is_first = is_first.float().reshape(B, -1).squeeze(-1)
        else:
            is_first = torch.zeros(B, device=self.device)

        # Initialize state on first call or episode reset
        if state is None:
            state = self.model.dynamics_model.initial(batch_size=B, seq_length=1, device=self.device)
            state["prev_action"] = torch.zeros(B, 1, self.model.dynamics_model.num_actions, device=self.device)
        else:
            state = tree_map(lambda x: torch.as_tensor(x).to(self.device) if x is not None else None, state)
            if state["prev_action"].dim() == 2:
                state["prev_action"] = state["prev_action"].unsqueeze(1)

        with torch.no_grad():
            # Delegate to UnifiedAgent.act which handles strategy switching
            action, new_state = self.model.act(obs, is_first, sample=(mode in ("train", "explore")))
            
        # Convert to numpy for env (maintain batch dimension B, strip sequence dimension L=1)
        # Sequence dimension is usually dim 1: (B, 1, ...). If len == 3, squeeze(1). 
        if action.dim() == 3:
            action = action.squeeze(1)
        elif action.dim() == 1:
            action = action.unsqueeze(0)
            
        action_np = action.cpu().numpy()
        state_np = tree_map(lambda x: x.cpu().numpy() if x is not None else None, new_state)
        return {"action": action_np}, state_np

    def train(self, data, state=None):
        """Train on a batch from replay buffer."""
        self.model.train()
        data = self._preprocess_obs(data)

        # Separate sensor data from control signals
        s = {k: v for k, v in data.items()
             if k not in ("action", "reward", "is_first", "is_last", "is_terminal",
                          "collision", "id", "env_action")}
        a = data["action"]
        r = data["reward"]
        d = 1.0 - data.get("is_terminal", torch.zeros_like(r)).float()
        f = data["is_first"].float()

        # Ensure correct dims: (B, L, 1)
        if r.dim() == 2: r = r.unsqueeze(-1)
        if d.dim() == 2: d = d.unsqueeze(-1)
        if f.dim() == 2: f = f.unsqueeze(-1)

        inputs = (s, a, r, d, f)

        # 1. World Model Training Step
        wm_loss_dict, wm_metrics, _ = self.model.world_model.train_step(
            inputs, inputs, self.model.config.precision, None, 1, 1, False)
        
        # 2. Intrinsic Reward / Exploration Training
        if self.expl is not None:
            # We need features and next states for Disag
            # These are computed during wm_loss calculation and cached in self.model
            # Or we can re-compute briefly if needed.
            # For simplicity, we use the detached posts from the model.
            with torch.no_grad():
                posts = self.model.detached_posts # (B*L, 1, D)
                feats = self.model.dynamics_model.get_feat(posts)
                
                # Targets are the next states
                # In (B, L, D) world, we shift by 1
                # posts['deter'] is (B, L, D)
                deter = posts['deter'].reshape(self.config.batch_size, -1, self.model.dynamics_model.hidden_size)
                targets = deter[:, 1:]
                input_feats = feats.reshape(self.config.batch_size, -1, feats.shape[-1])[:, :-1]
                input_actions = a[:, :-1]
            
            expl_metrics = self.expl.train_step(input_feats, input_actions, targets)
            wm_metrics.update(expl_metrics)
            
            # Add intrinsic reward to total reward for actor/critic
            intrinsic_reward = self.expl(input_feats, input_actions).unsqueeze(-1)
            # Clip or scale intrinsic reward
            intrinsic_reward = intrinsic_reward * self.config.get("expl_scale", 1.0)
            r[:, :-1] += intrinsic_reward

        # 3. Actor & Critic Training
        act_loss_dict, act_metrics, _ = self.model.actor_model.train_step(
            inputs, inputs, self.model.config.precision, None, 1, 1, False)
        crit_loss_dict, crit_metrics, _ = self.model.critic_model.train_step(
            inputs, inputs, self.model.config.precision, None, 1, 1, False)

        self.model.update_target_networks()
        self._updates += 1

        # 4. Consolidate and group metrics to eliminate duplicates
        metrics = {}
        
        def add_grouped_metrics(src, group_name):
            if not src: return
            for k, v in src.items():
                if k in ("loss", "total"):
                    metrics[f"{group_name}/loss"] = v
                elif k.startswith("loss_"):
                    # e.g., loss_stoch -> wm/stoch
                    sub_key = k[5:]
                    if sub_key != group_name: # avoid actor/actor if redundant
                        metrics[f"{group_name}/{sub_key}"] = v
                else:
                    metrics[f"{group_name}/{k}"] = v

        add_grouped_metrics(wm_loss_dict, "wm")
        add_grouped_metrics(act_loss_dict, "actor")
        add_grouped_metrics(crit_loss_dict, "critic")
        
        # Add auxiliary metrics (grad norms, model-specific stats)
        for src in [wm_metrics, act_metrics, crit_metrics]:
            for k, v in src.items():
                if k not in metrics:
                    # If it's already grouped (has /), keep it, else put in stats/
                    key = k if "/" in k else f"stats/{k}"
                    metrics[key] = v

        return {}, state, metrics

    def report(self, data):
        return self.model.report(data)

    def dataset(self, generator):
        """Wrap replay generator into batched iterator."""
        batch_size = self.config.get("batch_size", 4)
        def batched():
            gen = generator()
            while True:
                try:
                    batch = [next(gen) for _ in range(batch_size)]
                    yield {k: np.stack([b[k] for b in batch]) for k in batch[0].keys()}
                except StopIteration:
                    break
        return batched()

    def save(self):
        # Force state_dict to CPU before pickling to avoid GPU spikes on load
        cpu_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        return {"model": cpu_state_dict, "updates": self._updates}

    def load(self, data):
        import sys
        print("\n" + "="*60, file=sys.stderr)
        print("DEBUG: WorldModelAgent.load()", file=sys.stderr)
        print("="*60, file=sys.stderr)
        
        try:
            if not isinstance(data, dict):
                print(f"DEBUG: Data is not a dict, it is {type(data)}. Attempting to unpack...", file=sys.stderr)
                try:
                    data = pack.unpack(data)
                except Exception as e:
                    print(f"DEBUG: Unpack failed: {e}", file=sys.stderr)
                    return

            loaded_state_dict = data.get("model", data)
            current_state_dict = self.model.state_dict()
            
            filtered_state_dict = {}
            for k, v in loaded_state_dict.items():
                if k not in current_state_dict:
                    continue
                
                # Get shapes reliably
                if hasattr(v, "shape"):
                    v_shape = tuple(v.shape)
                elif isinstance(v, (list, tuple)):
                    v_shape = tuple(torch.tensor(v).shape)
                else:
                    print(f"DEBUG: Key {k} has no shape attribute ({type(v)})", file=sys.stderr)
                    continue
                    
                curr_shape = tuple(current_state_dict[k].shape)
                
                if v_shape == curr_shape:
                    if isinstance(v, torch.Tensor):
                        filtered_state_dict[k] = v.to(self.device)
                    else:
                        filtered_state_dict[k] = torch.from_numpy(v).to(self.device)
                else:
                    print(f"DEBUG: Shape mismatch for {k}: checkpoint {v_shape} != model {curr_shape}. Skipping.", file=sys.stderr)

            # Load the filtered state dict
            msg = self.model.load_state_dict(filtered_state_dict, strict=False)
            print(f"DEBUG: Load complete. Mismatched shapes handled. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}", file=sys.stderr)
            
            self._updates = data.get("updates", 0)
            self.model.to(self.device)
            
        except Exception as e:
            print(f"DEBUG: CRITICAL ERROR IN LOAD: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            
        print("="*60 + "\n", file=sys.stderr)
        sys.stderr.flush()


def tree_map(fn, tree):
    """Recursively apply fn to all leaves of a nested dict/list/tuple."""
    if isinstance(tree, dict):
        return {k: tree_map(fn, v) for k, v in tree.items()}
    elif isinstance(tree, (list, tuple)):
        return type(tree)(tree_map(fn, v) for v in tree)
    else:
        return fn(tree)
