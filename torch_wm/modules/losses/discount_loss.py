import torch
from torch_wm.core.base import BaseLoss
from torch_wm.core.registry import ModuleRegistry

@ModuleRegistry.register("discount")
class DiscountLoss(BaseLoss):
    def __init__(self, config=None, weight=1.0):
        super().__init__(config=config, weight=weight)

    def name(self): return "discount"
    def compute(self, model_outputs, batch, **kwargs):
        model_discounts = model_outputs["model_discounts"]
        dones = batch["dones"]

        # Ensure it is exactly (B, L, 1) and cast to float for arithmetic
        if dones.dim() == 2:
            dones = dones.unsqueeze(-1).float()
        else:
            dones = dones.float()
            
        discounts = (1.0 - dones)
        loss = - model_discounts.log_prob(discounts.detach()).mean()
        return loss
