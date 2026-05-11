import torch
import pytest
from torch_wm.models.wm_agent import WMAgent
from torch_wm.modules.losses.joint_embedding_loss import JointEmbeddingLoss
from torch_wm.modules.regularizers.sigreg import SIGRegLoss
from torch_wm.core.manager import LossManager

def test_sigreg_logic():
    """Verify SIGReg computes Epps-Pulley and allows gradients."""
    config = {"enabled": True, "weight": 1.0, "num_proj": 128}
    sigreg = SIGRegLoss(config=config)
    
    # Mock latents (B, L, D)
    B, L, D = 4, 8, 32
    latents = torch.randn(B, L, D, requires_grad=True)
    model_outputs = {"latent": latents}
    
    loss = sigreg.compute(model_outputs, {})
    assert loss.item() >= 0
    
    # Backward pass
    loss.backward()
    assert latents.grad is not None
    assert not torch.all(latents.grad == 0)

def test_jepa_loss_logic():
    """Verify JEPA aligns priors with detached posts."""
    config = {"enabled": True, "weight": 10.0, "sim_weight": 1.0, "pred_dim": 384, "target_dim": 384}
    jepa = JointEmbeddingLoss(config=config)
    
    B, L, D = 2, 4, 384
    # Mock model outputs
    # priors are predictions (require grad)
    # posts are actuals (require grad, but jepa should detach them)
    priors = {"stoch": torch.randn(B, L, D, requires_grad=True)}
    posts = {"stoch": torch.randn(B, L, D, requires_grad=True)}
    
    # In carTwister, latent is usually the posterior feat
    latent = torch.randn(B, L, D, requires_grad=True)
    
    model_outputs = {
        "priors": priors,
        "posts": posts,
        "latent": latent,
        "feats": torch.randn(B, L, D) # should be used as fallback if tssm not provided
    }
    
    # We need a mock tssm that converts state to features
    class MockTSSM:
        def get_feat(self, state):
            return state["stoch"]
            
    loss = jepa.compute(model_outputs, {}, tssm=MockTSSM())
    
    assert loss.item() >= 0
    loss.backward()
    
    # Gradients should flow to priors
    assert priors["stoch"].grad is not None
    
    # Gradients should NOT flow to latent (since it's detached as target)
    assert latent.grad is None or torch.all(latent.grad == 0)

def test_full_model_jepa_gradients():
    """Verify gradients flow from JEPA through the dynamics model."""
    from torch_wm.utils.yaml_config import load_yaml_config
    import os
    
    # Load default config
    config_path = "/home/quynhthu/Documents/world-model/carTwister/torch_wm/rl/config/twister.yaml"
    config = load_yaml_config(config_path)["defaults"]
    
    # Override for JEPA testing
    config["modules"]["losses"]["joint_embedding"]["enabled"] = True
    config["modules"]["losses"]["joint_embedding"]["weight"] = 10.0
    config["modules"]["losses"]["reconstruction"]["enabled"] = False
    config["modules"]["losses"]["sigreg"]["enabled"] = True
    config["modules"]["losses"]["sigreg"]["weight"] = 1.0
    
    # Mock obs_space for the test
    class SpaceStub:
        def __init__(self, s): self.shape = s
    obs_space = {
        "camera": SpaceStub((3, 64, 64)),
        "birdeye_wpt": SpaceStub((3, 64, 64))
    }
    
    agent = WMAgent(env_name="test_env", override_config=config, skip_env=True, obs_space=obs_space)
    
    # Create manual batch with correct shapes for WorldModel
    # WorldModel expects (B, L, ...)
    B, L = 2, 5
    obs = {
        "camera": torch.zeros((B, L, 3, 64, 64)),
        "birdeye_wpt": torch.zeros((B, L, 3, 64, 64))
    }
    actions = torch.zeros((B, L, 15))
    rewards = torch.zeros((B, L))
    dones = torch.zeros((B, L))
    is_firsts = torch.zeros((B, L))
    is_firsts[:, 0] = 1.0 # First step is always first
    
    # Training step
    agent.train()
    # Call forward directly on the world model to check loss computation and gradient flow
    # WorldModel forward expects a tuple/list of inputs: (s, a, r, d, f)
    loss = agent.world_model((obs, actions, rewards, dones, is_firsts))
    loss.backward()
    
    # Metrics are stored in loss_manager or agent.world_model.added_losses
    # Note: WorldModel.forward already calls loss_manager.get_metrics() and puts them in added_metrics
    metrics = agent.world_model.added_metrics
    print(f"Metrics: {metrics.keys()}")
    
    # Check for keys containing our loss names (using relaxation for prefixing)
    assert any("joint_embedding" in k for k in metrics.keys()), f"No JEPA metrics in {metrics.keys()}"
    assert any("sigreg" in k for k in metrics.keys()), f"No SIGReg metrics in {metrics.keys()}"
    
    # Check that dynamics model has gradients
    has_grad = False
    for name, p in agent.dynamics_model.named_parameters():
        if p.grad is not None:
            has_grad = True
            break
    assert has_grad, "Dynamics model should have gradients from JEPA loss"
    
    # Check that JEPA projectors have gradients
    jepa_loss = agent.world_model.loss_manager.active_losses[1] # joint_embedding is second after kl
    assert jepa_loss.predictor_projector[0].weight.grad is not None, "JEPA predictor projector should have gradients"
    assert jepa_loss.target_projector[0].weight.grad is not None, "JEPA target projector should have gradients"

if __name__ == "__main__":
    test_sigreg_logic()
    test_jepa_loss_logic()
    test_full_model_jepa_gradients()
    print("All tests passed!")
