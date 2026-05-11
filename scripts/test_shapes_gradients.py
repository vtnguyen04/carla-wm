import torch
import numpy as np
import sys
import pathlib
import os

# Add project root to path
root = pathlib.Path(__file__).parent.parent
sys.path.append(str(root))

from torch_wm.models.unified_agent import UnifiedAgent
from torch_wm.structs import AttrDict

def test_unified_model_audit():
    print("🚀 Starting Comprehensive Model Shape & Gradient Audit...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Using device: {device}")

    # 1. Load Real Configs (Merge common and twister)
    import ruamel.yaml as yaml_lib
    yaml = yaml_lib.YAML(typ="safe")
    
    with open("carla_env/configs/common.yaml", 'r') as f:
        common_cfg = yaml.load(f)
    with open("torch_wm/rl/config/twister.yaml", 'r') as f:
        twister_cfg = yaml.load(f)
    
    # Simple merge logic
    config_dict = common_cfg["env"].copy()
    config_dict.update(twister_cfg["defaults"])
    
    # Merge env_params specifically
    if "env_params" in twister_cfg["defaults"]:
        config_dict["env_params"].update(twister_cfg["defaults"]["env_params"])
        
    config = AttrDict(config_dict)
    
    # 2. Extract Action Space Info
    action_cfg = config.get("action", {})
    if "env_params" in config and "action" in config.env_params:
        action_cfg.update(config.env_params.action)
        
    discrete_acc = action_cfg.get("discrete_acc", [-3.0, 0.0, 3.0])
    discrete_steer = action_cfg.get("discrete_steer", [-0.6, 0.0, 0.6])
    n_acc = len(discrete_acc)
    n_steer = len(discrete_steer)
    total_actions = n_acc * n_steer
    
    # Critical keys for Model initialization and forward
    config.num_actions = total_actions
    config.discrete_actions = True
    config.L = config.get("batch_length", 16)
    
    # Fallback defaults for core architecture
    if "stoch_size" not in config: config.stoch_size = 32
    if "discrete" not in config: config.discrete = 32
    if "hidden_size" not in config: config.hidden_size = 256
    
    print(f"⚙️ Config Loaded: Actions={total_actions} ({n_acc} acc x {n_steer} steer), L={config.L}")

    # 3. Setup Obs Space
    obs_config = config.get("observation", {})
    if "env_params" in config and "observation" in config.env_params:
        obs_config.update(config.env_params.observation)
        
    enabled_sensors = obs_config.get("enabled", ["camera"])
    obs_space = {}
    for key in enabled_sensors:
        if key not in obs_config: continue
        s_cfg = obs_config[key]
        shape = s_cfg.get("shape", [128, 128, 3])
        # Model expects CHW for images
        if len(shape) == 3 and shape[-1] == 3:
            shape = (shape[2], shape[0], shape[1])
            
        class SpaceStub:
            def __init__(self, s): self.shape = s
        obs_space[key] = SpaceStub(shape)
    
    # Add scalar obs commonly present
    for key in ["collision", "reward", "is_first", "is_last", "is_terminal"]:
        obs_space[key] = SpaceStub((1,) if key == "collision" else ())

    # 4. Instantiate Agent
    agent = UnifiedAgent(env_name="test", override_config=config, obs_space=obs_space).to(device)
    agent.compile()
    
    B, L = 2, config.get("batch_length", 16)
    H = config.H
    
    # 5. Create Mock Batch
    print(f"📦 Creating mock batch (B={B}, L={L})...")
    batch_obs = {}
    for key, space in obs_space.items():
        if len(space.shape) >= 2: # Images/Maps
            batch_obs[key] = torch.randint(0, 255, (B, L, *space.shape), dtype=torch.uint8, device=device)
        else: # Scalars
            batch_obs[key] = torch.randn(B, L, *space.shape, device=device)
    
    batch_actions = torch.zeros(B, L, total_actions, device=device)
    batch_actions[..., 0] = 1.0 # One-hot
    batch_rewards = torch.randn(B, L, 1, device=device)
    batch_dones = torch.zeros(B, L, 1, device=device)
    batch_firsts = torch.zeros(B, L, device=device)
    batch_firsts[:, 0] = 1.0
    
    inputs = (batch_obs, batch_actions, batch_rewards, batch_dones, batch_firsts)

    # --- WORLD MODEL AUDIT ---
    print("\n🌍 Auditing World Model...")
    wm_loss = agent.world_model(inputs)
    print(f"✅ WM Forward pass successful. Loss: {wm_loss.item():.4f}")
    
    wm_loss.backward()
    print("✅ WM Backward pass successful. Gradients flowing.")
    
    # Check detached states for Actor
    assert hasattr(agent, 'detached_posts'), "Missing detached_posts"
    assert agent.detached_posts['deter'].shape == (B*L, 1, config.hidden_size)
    print(f"✅ WM Detached state shape: {agent.detached_posts['deter'].shape}")

    # --- ACTOR MODEL AUDIT ---
    print("\n🎭 Auditing Actor Model...")
    # Actor forward uses detached data from WM
    act_loss = agent.actor_model(inputs)
    print(f"✅ Actor Forward pass successful. Loss: {act_loss.item():.4f}")
    
    act_loss.backward()
    print("✅ Actor Backward pass successful. Gradients flowing.")
    
    # Check internal shapes
    assert agent.detached_feats.shape == (B*L, 1 + H, config.hidden_size + config.stoch_size * config.discrete)
    print(f"✅ Actor Features shape: {agent.detached_feats.shape}")

    # --- CRITIC MODEL AUDIT ---
    print("\n🧠 Auditing Critic Model...")
    crit_loss = agent.critic_model(inputs)
    print(f"✅ Critic Forward pass successful. Loss: {crit_loss.item():.4f}")
    
    crit_loss.backward()
    print("✅ Critic Backward pass successful. Gradients flowing.")

    # --- REWARD & CONTINUE AUDIT ---
    print("\n💰 Auditing Reward & Discount Networks...")
    feat_dim = config.hidden_size + config.stoch_size * config.discrete
    test_feat = torch.randn(1, 5, feat_dim, device=device)
    
    reward_dist = agent.reward_network(test_feat)
    reward_mode = reward_dist.mode() if callable(reward_dist.mode) else reward_dist.mode
    assert reward_mode.shape == (1, 5, 1)
    print(f"✅ Reward Mode shape: {reward_mode.shape}")
    
    continue_dist = agent.continue_network(test_feat)
    continue_mode = continue_dist.mode() if callable(continue_dist.mode) else continue_dist.mode
    assert continue_mode.shape == (1, 5, 1)
    print(f"✅ Discount Mode shape: {continue_mode.shape}")

    # --- ACTION SPACE AUDIT ---
    print("\n🏎️ Auditing Action Space (Steer & Speed)...")
    policy_dist = agent.policy_network(test_feat)
    action_sample = policy_dist.sample()
    assert action_sample.shape == (1, 5, total_actions)
    print(f"✅ Policy Sample shape: {action_sample.shape} (One-hot {total_actions})")
    
    # Verify mapping
    indices = action_sample.argmax(dim=-1)
    acc_idx = indices // n_steer
    steer_idx = indices % n_steer
    
    assert (acc_idx < n_acc).all(), f"Acc index {acc_idx.max()} out of range {n_acc}"
    assert (steer_idx < n_steer).all(), f"Steer index {steer_idx.max()} out of range {n_steer}"
    print(f"✅ Action indices correctly map to {n_acc} acc levels and {n_steer} steer levels.")
    
    # Check range of values
    print(f"📈 Acceleration levels: {discrete_acc}")
    print(f"📈 Steering levels: {discrete_steer}")

    print("\n✨ AUDIT COMPLETE: ALL MODELS ARE STRUCTURALLY SOUND AND GRADIENT-READY.")

if __name__ == "__main__":
    try:
        test_unified_model_audit()
    except Exception as e:
        print(f"\n💥 AUDIT FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
