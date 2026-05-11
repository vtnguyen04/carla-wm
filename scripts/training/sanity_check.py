
import torch
import numpy as np
from torch_wm.models.wm_agent import WMAgent
from torch_wm.structs import AttrDict

def sanity_check():
    print("Starting Training Sanity Check...")
    
    # 1. Setup Config (Minimal)
    config = AttrDict({
        "env_name": "test_sanity",
        "stoch_size": 16,
        "discrete": 16,
        "hidden_size": 256,
        "num_blocks_trans": 1,
        "att_context_left": 8,
        "image_size": (64, 64),
        "patch_size": 16,
        "tubelet_size": 2,
        "batch_size": 2,
        "batch_length": 4,
        'precision': 'float32',
        'jit': False,
        'compile': False,
        'num_actions': 3,
        'obs': {'camera': {'shape': (3, 64, 64), 'tssm': True}},
        'H': 5, # Imaging horizon
        'gamma': 0.99,
        'lambda_td': 0.95,
        'target_value_reg': False,
        'actor_ent': 1e-4,
        'slow_critic_update': 1,
        'slow_critic_fraction': 0.02,
        'discrete_steer': [-0.6, 0.0, 0.6],
        'discrete_acc': [-3.0, 0.0, 3.0],
        "model_opt": {"lr": 1e-4, "eps": 1e-8, "wd": 0.0, "clip": 100.0, "warmup": 0, "decay_steps": 1000, "min_lr": 1e-5},
        "actor_opt": {"lr": 8e-5, "eps": 1e-5, "wd": 0.0, "clip": 100.0, "warmup": 0, "decay_steps": 1000, "min_lr": 8e-6},
        "critic_opt": {"lr": 8e-5, "eps": 1e-5, "wd": 0.0, "clip": 100.0, "warmup": 0, "decay_steps": 1000, "min_lr": 8e-6},
        "env_params": {
            "observation": {
                "camera": {"shape": (3, 64, 64), "tssm": True}
            }
        },
        "modules": {"losses": {"kl": {"enabled": True}, "reconstruction": {"enabled": True}, "cpc": {"enabled": False}}},
        "loss_scales": {"image": 1.0, "vector": 1.0, "reward": 1.0, "cont": 1.0, "dyn": 0.5, "rep": 0.1, "actor": 1.0, "critic": 1.0, "slowreg": 1.0}
    })

    # 2. Instantiate Agent
    agent = WMAgent(env_name="test_sanity", override_config=config, skip_env=True)
    agent.to("cpu")
    agent.compile()
    
    print("Agent compiled successfully.")

    # 3. Create Mock Data
    B, L = config.batch_size, config.batch_length
    inputs = {
        "camera": torch.randn(B, L, 3, 64, 64).clamp(0, 1), # Raw pixels [0, 1]
        "action": torch.zeros(B, L, agent.num_actions),
        "reward": torch.zeros(B, L),
        "discount": torch.ones(B, L),
        "is_first": torch.zeros(B, L)
    }
    inputs["is_first"][:, 0] = 1.0 # First step is first
    
    # Targets usually include the ground truth observations for reconstruction
    targets = {
        "camera": inputs["camera"].clone(),
        "reward": inputs["reward"].clone(),
        "discount": inputs["discount"].clone()
    }
    
    # 4. Run Train Step
    print("Running train_step...")
    # Use float32 and no scaler for CPU sanity check
    losses, metrics, _ = agent.train_step(
        inputs, targets={}, precision=torch.float32, grad_scaler=None, 
        accumulated_steps=1, acc_step=0, eval_training=False
    )
    
    print("\nTraining Step Results:")
    if isinstance(losses, dict):
        for group, loss_val in losses.items():
            if isinstance(loss_val, dict) and 'total' in loss_val:
                print(f"  {group.upper()} Total Loss: {loss_val['total']:.4f}")
            else:
                print(f"  {group.upper()} Loss: {loss_val}")
    
    print("\nMetrics:")
    for k, v in metrics.items():
        if isinstance(v, (float, int, torch.Tensor)):
            val = v.item() if isinstance(v, torch.Tensor) else v
            print(f"  {k}: {val:.4f}")

    print("\nSanity Check PASSED! Model is trainable and gradients are flowing.")

if __name__ == "__main__":
    sanity_check()
