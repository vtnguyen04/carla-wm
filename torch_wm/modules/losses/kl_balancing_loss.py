import torch
from torch_wm.core.registry import ModuleRegistry
from torch_wm.core.base import BaseLoss
import torch.distributions as tfd
from torch_wm.distributions import OneHotDist

@ModuleRegistry.register('kl_balancing')
class KLBalancingLoss(BaseLoss):
    def name(self):
        return "kl_balancing"

    def __init__(self, config, weight):
        super().__init__(config, weight)
        self.free_bits = getattr(config, 'free_bits', 1.0)
        self.dyn_scale = getattr(config, 'dyn_scale', 0.2)
        self.rep_scale = getattr(config, 'rep_scale', 0.8)
        self.discrete = getattr(config, 'discrete', True)

    def get_dist(self, stats):
        if self.discrete:
            return tfd.Independent(OneHotDist(logits=stats['logits']), 1)
        else:
            return tfd.Independent(tfd.Normal(stats['mean'], stats['std']), 1)

    def compute(self, model_outputs, batch, **kwargs):
        """
        Compute the KL Balancing Loss between post (posterior) and prior.
        """
        posts = model_outputs['posts']
        priors = model_outputs['priors']
        mask = model_outputs.get('mask', None)

        post_dist = self.get_dist(posts)
        prior_dist = self.get_dist(priors)

        # Dynamics loss: Train prior towards post
        # sg(post) || prior
        post_sg_dist = self.get_dist({k: v.detach() for k, v in posts.items() if isinstance(v, torch.Tensor)})
        dyn_loss = torch.distributions.kl.kl_divergence(post_sg_dist, prior_dist)
        dyn_loss = torch.max(dyn_loss, torch.full_like(dyn_loss, self.free_bits))

        # Representation loss: Train post towards prior
        # post || sg(prior)
        prior_sg_dist = self.get_dist({k: v.detach() for k, v in priors.items() if isinstance(v, torch.Tensor)})
        rep_loss = torch.distributions.kl.kl_divergence(post_dist, prior_sg_dist)
        rep_loss = torch.max(rep_loss, torch.full_like(rep_loss, self.free_bits))

        loss = self.dyn_scale * dyn_loss + self.rep_scale * rep_loss

        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / mask.sum()
        else:
            loss = loss.mean()

        # Update metrics
        self.update_metrics({
            "dyn_loss": dyn_loss.mean().item(),
            "rep_loss": rep_loss.mean().item(),
            "kl_total": loss.item()
        })

        return loss
