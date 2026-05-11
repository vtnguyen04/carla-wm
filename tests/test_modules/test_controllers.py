import pytest
import numpy as np

from torch_wm.modules.controllers.pure_pursuit import PurePursuitController
from torch_wm.modules.controllers.stanley_controller import StanleyController
from torch_wm.modules.controllers.pid_controller import PIDController

def test_pure_pursuit():
    controller = PurePursuitController(wheelbase=2.96, max_steer_rad=1.22)
    state = {'x': 0.0, 'y': 0.0, 'yaw': 0.0, 'speed': 5.0}
    
    # Simple straight trajectory
    trajectory = np.array([
        [0.0, 0.0],
        [5.0, 0.0],
        [10.0, 0.0]
    ])
    
    control = controller.compute(state, trajectory, target_speed=8.0)
    assert 'steer' in control
    assert 'throttle' in control
    assert 'brake' in control
    
def test_stanley_controller():
    controller = StanleyController(wheelbase=2.96, max_steer_rad=1.22)
    state = {'x': 0.0, 'y': 1.0, 'yaw': 0.0, 'speed': 5.0} # 1m offset
    
    # Simple straight trajectory
    trajectory = np.array([
        [0.0, 0.0],
        [5.0, 0.0],
        [10.0, 0.0]
    ])
    
    control = controller.compute(state, trajectory, target_speed=8.0)
    assert 'steer' in control
    assert 'throttle' in control
    assert 'brake' in control

def test_pid_controller():
    controller = PIDController()
    error1 = controller.run(1.0)
    error2 = controller.run(0.5)
    assert isinstance(error1, float)
    assert isinstance(error2, float)
