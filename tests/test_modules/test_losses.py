import pytest
import torch
from torch_wm.modules.losses.kl_loss import KLLoss
from torch_wm.modules.losses.reconstruction_loss import ReconstructionLoss
from torch_wm.modules.losses.reward_loss import RewardLoss
from torch_wm.modules.losses.joint_embedding_loss import JointEmbeddingLoss
from torch_wm.modules.losses.vcreg_loss import VCRegLoss

# Mock TSSM for get_dist
class MockTSSM:
    def get_dist(self, state):
        # Return a simple Normal dist to calculate KL divergence
        loc = state["stoch"]
        return torch.distributions.Normal(loc, torch.ones_like(loc))

def test_kl_loss():
    kl_loss = KLLoss(config={"free_nats": 0.5, "prior_scale": 1.0, "post_scale": 1.0})
    
    B, T = 2, 3
    posts = {"stoch": torch.randn(B, T, 32)}
    priors = {"stoch": torch.randn(B, T, 32)}
    
    model_outputs = {
        "posts": posts,
        "priors": priors
    }
    
    loss = kl_loss.compute(model_outputs, batch={}, tssm=MockTSSM())
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0 # scalar
    assert loss.item() >= 0.0

class MockDist:
    def __init__(self, value):
        self.value = value
    def log_prob(self, target):
        # Dummy log prob is just negative MSE
        return -((self.value - target) ** 2)

def test_reconstruction_loss():
    recon_loss = ReconstructionLoss(config={"cnn_keys": ["camera"]})
    
    B, T = 2, 3
    model_outputs = {
        "states_rec_dist": {
            "camera": MockDist(torch.ones(B, T, 3, 64, 64))
        }
    }
    batch = {
        "states": {
            "camera": torch.zeros(B, T, 3, 64, 64)
        }
    }
    
    loss = recon_loss.compute(model_outputs, batch)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    # negative log_prob is (1**2).mean() = 1.0
    assert torch.isclose(loss, torch.tensor(1.0))
    
def test_reconstruction_loss_missing_keys():
    recon_loss = ReconstructionLoss()
    loss = recon_loss.compute(model_outputs={}, batch={})
    assert loss.item() == 0.0

def test_reward_loss():
    reward_loss = RewardLoss()
    
    B, T = 2, 3
    model_outputs = {
        "model_rewards": MockDist(torch.ones(B, T, 1))
    }
    batch = {
        "rewards": torch.zeros(B, T) # check dim expansion
    }
    
    loss = reward_loss.compute(model_outputs, batch)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert torch.isclose(loss, torch.tensor(1.0))

def test_joint_embedding_loss():
    config = {"sim_weight": 1.0, "var_weight": 0.1}
    loss_mod = JointEmbeddingLoss(config)
    assert loss_mod.name() == "joint_embedding"
    
    # Mock data
    post = {"deter": torch.randn(2, 4, 32)}
    prior = {"deter": torch.randn(2, 4, 32)}
    latent = torch.randn(2, 4, 64) # Match feats size
    feats = torch.randn(2, 4, 64)
    model_outputs = {"posts": post, "priors": prior, "latent": latent, "feats": feats}
    
    loss_val = loss_mod.compute(model_outputs, {})
    assert loss_val > 0
    metrics = loss_mod.get_metrics()
    assert "joint_embedding/loss_je_sim" in metrics
    assert "joint_embedding/loss_je_var" in metrics

def test_vcreg_loss():
    config = {"var_weight": 1.0, "cov_weight": 1.0}
    loss_mod = VCRegLoss(config)
    assert loss_mod.name() == "vcreg"
    
    # Mock data
    post = {"deter": torch.randn(4, 32)} # B=4, D=32
    model_outputs = {"posts": post}
    
    loss_val = loss_mod.compute(model_outputs, {})
    assert loss_val > 0
    metrics = loss_mod.get_metrics()
    assert "vcreg/loss_vcreg_var" in metrics
    assert "vcreg/loss_vcreg_cov" in metrics

from torch_wm.modules.losses.cpc_loss import CPCLoss
from torch_wm.modules.losses.discount_loss import DiscountLoss
from torch_wm.modules.losses.kl_balancing_loss import KLBalancingLoss

def test_cpc_loss():
    config = {"cpc_weight": 1.0}
    loss_mod = CPCLoss(config)
    
    B, T, D = 2, 4, 32
    model_outputs = {
        "latent": torch.randn(B, T, D),
        "feats": torch.randn(B, T, D)
    }
    
    # Mock contrastive network
    class MockCPC(torch.nn.Module):
        def forward(self, f, l):
            return torch.randn(f.shape[0], f.shape[1], 16), torch.randn(l.shape[0], l.shape[1], 16)
    
    contrastive_network = torch.nn.ModuleList([MockCPC()])
    
    loss_val = loss_mod.compute(model_outputs, {}, contrastive_network=contrastive_network)
    assert loss_val >= 0
    assert "cpc/cpc_accuracy" in loss_mod.get_metrics()

def test_discount_loss():
    loss_mod = DiscountLoss()
    
    B, T = 2, 4
    model_outputs = {
        "model_discounts": MockDist(torch.ones(B, T, 1))
    }
    batch = {
        "dones": torch.zeros(B, T)
    }
    
    loss_val = loss_mod.compute(model_outputs, batch)
    assert loss_val >= 0
    assert torch.isclose(loss_val, torch.tensor(0.0), atol=1e-5)

def test_kl_balancing_loss():
    # Fix: needs 2 args (config, weight)
    loss_mod = KLBalancingLoss(config={"free_nats": 0.5, "dyn_scale": 0.5, "rep_scale": 0.1}, weight=1.0)
    
    B, T, D = 2, 4, 32
    # Fix: needs 'logits' because get_dist uses OneHotDist
    model_outputs = {
        "posts": {"logits": torch.randn(B, T, D)},
        "priors": {"logits": torch.randn(B, T, D)}
    }
    
    loss_val = loss_mod.compute(model_outputs, {})
    assert loss_val >= 0
    metrics = loss_mod.get_metrics()
    assert "kl_balancing/dyn_loss" in metrics
    assert "kl_balancing/rep_loss" in metrics
