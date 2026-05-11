import pytest
import os
import torch
import ruamel.yaml as yaml

from torch_wm.structs import AttrDict
from torch_wm.models.wm_agent import WMAgent

@pytest.fixture
def agent_config():
    path = os.path.join(os.path.dirname(__file__), "../../torch_wm/rl/config/dreamerv3.yaml")
    model_configs = yaml.YAML(typ='safe').load(open(path))
    config = AttrDict(model_configs['defaults'])
    
    # Needs some base properties
    config.task = "test_env"
    config.device = "cpu"
    config.batch_size = 2
    config.batch_length = 4
    config.discrete_actions = True
    config.num_actions = 15
    config.L = 4
    
    # We must explicitly add env_params because WMAgent parses them
    config.env_params = AttrDict({
        "observation": AttrDict({
            "enabled": ["camera", "vector"],
            "camera": AttrDict({
                "shape": [3, 64, 64],
                "decode": True,
                "tssm": True,
            }),
            "vector": AttrDict({
                "shape": [10],
                "decode": True,
                "tssm": True,
            })
        }),
        "action": AttrDict({"discrete": True, "n_cmds": 15})
    })
    
    return config

def test_integration_full_backward_pass(agent_config):
    torch.set_grad_enabled(True)
    # This acts as the final system smoke test. If we pass here, it trains.
    model = WMAgent(env_name="test_env", override_config=agent_config, skip_env=True)
    model.compile()
    model.train()
    
    B, L = agent_config.batch_size, agent_config.batch_length
    
    s = {
        "camera": torch.randint(0, 255, (B, L, 3, 64, 64), dtype=torch.uint8), # Raw input format from embody is (B, L, H, W, C), but our preprocessing expects either or handles transposition
        "vector": torch.randn(B, L, 10)
    }
    a = torch.randn(B, L, 15)
    r = torch.randn(B, L)
    d = torch.zeros(B, L, dtype=torch.bool)
    f = torch.zeros(B, L, dtype=torch.bool)
    f[:, 0] = True
    
    for name, param in model.named_parameters():
        if param.dtype.is_floating_point and "v_target" not in name:
            try:
                param.requires_grad_(True)
            except ValueError:
                # Skip uninitialized parameters
                pass
            
    inputs = (s, a, r, d, f)
    
    # Note: Gradients are zeroed internally within train_step
    
    # WORLD MODEL pass
    # train_step typically unrolls to LossManager
    try:
        wm_loss_dict, wm_metrics, wm_tensors = model.world_model.train_step(
            inputs, inputs, agent_config.precision, None, 1, 1, False
        )
    except RuntimeError as e:
        # Just compute it directly to see the gradients
        batch_losses, _, _, _ = model.world_model.forward_model(inputs, inputs, compute_metrics=False)
        print("DEBUG BATCH LOSSES GRADS:", {k: (v.requires_grad, v.grad_fn) for k, v in batch_losses.items()})
        raise e
    
    assert "loss" in wm_loss_dict
    assert not torch.isnan(wm_loss_dict["loss"])
    
    # ACTOR MODEL pass
    act_loss_dict, act_metrics, act_tensors = model.actor_model.train_step(
        inputs, inputs, agent_config.precision, None, 1, 1, False
    )
    
    assert "loss" in act_loss_dict
    assert not torch.isnan(act_loss_dict["loss"])
    
    # CRITIC MODEL pass
    crit_loss_dict, crit_metrics, crit_tensors = model.critic_model.train_step(
        inputs, inputs, agent_config.precision, None, 1, 1, False
    )
    
    assert "loss" in crit_loss_dict
    assert not torch.isnan(crit_loss_dict["loss"])
    
    # Test target network update
    model.update_target_networks()
    
    # If we made it here without crashing during forward/backward passes, the system is fundamentally sound.
