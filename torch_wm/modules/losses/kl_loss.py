import torch
from torch_wm.core.base import BaseLoss
from torch_wm.core.registry import ModuleRegistry

@ModuleRegistry.register("kl")
class KLLoss(BaseLoss):
    def __init__(self, config=None, weight=1.0):
        super().__init__(config=config, weight=weight)
        self.free_nats = self.config.get("free_nats", 1.0)
        self.prior_scale = self.config.get("prior_scale", 0.5)
        self.post_scale = self.config.get("post_scale", 0.1)

    def name(self): return "kl"

    def compute(self, model_outputs, batch, **kwargs):
        posts = model_outputs["posts"]
        priors = model_outputs["priors"]
        tssm = kwargs.get("tssm")
        
        if tssm is None:
            return torch.tensor(0.0, device=posts["stoch"].device)

        # KL Calculation
        kl_prior = torch.distributions.kl.kl_divergence(
            tssm.get_dist({k: v if k == "hidden" else v.detach() for k, v in posts.items()}), 
            tssm.get_dist(priors)
        )
        kl_post = torch.distributions.kl.kl_divergence(
            tssm.get_dist(posts), 
            tssm.get_dist({k: v if k == "hidden" else v.detach() for k, v in priors.items()})
        )

        loss = self.prior_scale * torch.mean(torch.clip(kl_prior, min=self.free_nats)) + \
               self.post_scale * torch.mean(torch.clip(kl_post, min=self.free_nats))
        
        return loss
