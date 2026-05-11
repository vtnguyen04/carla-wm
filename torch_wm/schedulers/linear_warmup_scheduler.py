# PyTorch
import torch
import math

# NeuralNets
from torch_wm.schedulers.scheduler import Scheduler

class LinearWarmupScheduler(Scheduler):
    """Linear Warmup Scheduler.
    
    Linearly increases the learning rate from 0 to target_lr over warmup_steps,
    then holds it constant.
    """

    def __init__(self, target_lr, warmup_steps):
        super().__init__()
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps

    def get_val(self):
        # Warmup phase
        if self.model_step <= self.warmup_steps:
            return self.target_lr * (self.model_step / self.warmup_steps)
            
        # Constant phase
        return self.target_lr


class LinearWarmupCosineScheduler(Scheduler):
    """Linear Warmup + Cosine Annealing Scheduler.
    
    Phase 1 (step 0 → warmup_steps):
        Linearly ramp from 0 → target_lr
    
    Phase 2 (warmup_steps → total_steps):
        Cosine decay from target_lr → target_lr * min_lr_ratio
    
    This is the standard schedule for world models (DreamerV3, WMAgent).
    Cosine decay prevents the learning rate from staying too high,
    which causes oscillation and prevents fine-grained convergence.
    """

    def __init__(self, target_lr, warmup_steps, total_steps=1_000_000, min_lr_ratio=0.1):
        super().__init__()
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = target_lr * min_lr_ratio

    def get_val(self):
        step = self.model_step
        
        # Phase 1: Linear warmup
        if step <= self.warmup_steps:
            return self.target_lr * (step / max(self.warmup_steps, 1))
        
        # Phase 2: Cosine annealing
        progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        progress = min(progress, 1.0)  # Clamp at end
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.target_lr - self.min_lr) * cosine_decay
