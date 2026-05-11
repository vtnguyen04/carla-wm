import pytest
import math
from torch_wm.schedulers.linear_warmup_scheduler import LinearWarmupScheduler, LinearWarmupCosineScheduler
from torch_wm.schedulers.constant_scheduler import ConstantScheduler

def test_linear_warmup_scheduler():
    scheduler = LinearWarmupScheduler(target_lr=1.0, warmup_steps=10)
    scheduler.step() # step 1
    assert scheduler.get_val() == 0.1
    for _ in range(9):
        scheduler.step()
    assert scheduler.get_val() == 1.0 # step 10
    scheduler.step()
    assert scheduler.get_val() == 1.0 # step 11

def test_linear_warmup_cosine_scheduler():
    scheduler = LinearWarmupCosineScheduler(target_lr=1.0, warmup_steps=10, total_steps=20, min_lr_ratio=0.1)
    scheduler.step() # step 1
    assert abs(float(scheduler.get_val()) - 0.1) < 1e-4
    for _ in range(9):
        scheduler.step()
    assert abs(float(scheduler.get_val()) - 1.0) < 1e-4 # step 10
    for _ in range(10):
        scheduler.step()
    assert abs(float(scheduler.get_val()) - 0.1) < 1e-4 # step 20
    scheduler.step()
    assert abs(float(scheduler.get_val()) - 0.1) < 1e-4 # step 21
