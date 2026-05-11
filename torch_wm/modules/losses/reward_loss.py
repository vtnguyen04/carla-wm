import torch
from torch_wm.core.base import BaseLoss
from torch_wm.core.registry import ModuleRegistry

@ModuleRegistry.register("reward")
class RewardLoss(BaseLoss):
    def __init__(self, config=None, weight=1.0):
        super().__init__(config=config, weight=weight)
    
    def name(self): return "reward"
    def compute(self, model_outputs, batch, **kwargs):
        model_rewards = model_outputs["model_rewards"]
        rewards = batch["rewards"]
        
        # Ensure it is exactly (B, L, 1)
        if rewards.dim() == 2:
            rewards = rewards.unsqueeze(-1)
        elif rewards.dim() > 3:
            rewards = rewards.view(rewards.shape[0], rewards.shape[1], 1)
            
        # Loss computation
        loss = - model_rewards.log_prob(rewards.detach()).mean()

        return loss
