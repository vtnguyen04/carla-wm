import math
import time
from abc import abstractmethod
from typing import Dict, Tuple

import carla
import gym
import numpy as np
from gym import spaces

from carla_env.toolkit.utils import get_logger

from .toolkit import EnvMonitorLocalCV, EnvMonitorOpenCV, Observer, WorldManager

log = get_logger(log_dir=".", job_name="carla_base_env")


class CarlaBaseEnv(gym.Env):
    def __init__(self, config):
        self._config = config

        if self._config.display.get("no_monitor", False):
            self._monitor = None
        elif self._config.display.get("use_local_window", False):
            from .toolkit.monitor.pygame_monitor import EnvMonitorPygame

            self._monitor = EnvMonitorPygame(self._config)
        elif self._config.display.get("save_video", False):
            from .toolkit.monitor.monitor import EnvMonitorVideo

            self._monitor = EnvMonitorVideo(self._config)
        else:
            self._monitor = EnvMonitorOpenCV(self._config)

        self._world = WorldManager(self._config)
        self._world.on_reset(self.on_reset)
        self._world.on_step(self.on_step)
        self._observer = Observer(
            self._world, self._config.observation, self._config.display
        )

        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

        self.prev_steer = 0.0
        self.prev_acc = 0.0

        # self._image_keys = []
        # for key, space in self.observation_space.spaces.items():
        #     if isinstance(space, spaces.Box) and len(space.shape) == 3 and space.dtype == np.uint8:
        #         self._image_keys.append(key)

    @abstractmethod
    def on_reset(self) -> None:
        """
        Override this method to perform additional reset operations.
        Specifically, you can spawn actors and plan routes here.
        """
        pass

    @abstractmethod
    def apply_control(self, action) -> None:
        """
        Override this method to apply control to actors.
        This method will be called before the simulator ticks.
        """
        pass

    @abstractmethod
    def on_step(self) -> None:
        """
        Override this method to perform additional operations at each step.
        Specifically, you can update the planner and the route here.
        This method will be called after the simulator ticks.
        """
        pass

    @abstractmethod
    def reward(self) -> Tuple[float, Dict]:
        """
        Override this method to define the reward function.
        """
        pass

    @abstractmethod
    def get_terminal_conditions(self) -> Dict[str, bool]:
        """
        Override this method to define the terminal condition.
        If one of the keys in the returned dictionary gives True, the episode will be terminated.
        """
        pass

    def get_ego_vehicle(self) -> carla.Actor:
        """
        IPC OPTIMIZATION: Cache ego vehicle reference to avoid IPC calls
        """
        if hasattr(self, "_cached_ego") and self._cached_ego is not None:
            return self._cached_ego
        return self.ego

    def get_state(self) -> Dict:
        """Return the environment state. Implement this method to define the env state."""
        return self._state

    def _get_action_space(self):
        action_config = self._config.action
        if action_config.discrete:
            self.n_steer = len(action_config.discrete_steer)
            self.n_acc = len(action_config.discrete_acc)
            return spaces.Discrete(self.n_steer * self.n_acc)
        else:
            return spaces.Box(
                low=np.array(
                    [action_config.continuous_acc[0], action_config.continuous_steer[0]]
                ),
                high=np.array(
                    [action_config.continuous_acc[1], action_config.continuous_steer[1]]
                ),
                dtype=np.float32,
            )

    def _get_observation_space(self):
        return self._observer.get_observation_space()

    def reset(self):
        log.info("[CARLA] Reset environment")

        # Clear cached ego reference for map switching
        if hasattr(self, "_cached_ego"):
            del self._cached_ego

        self._observer.destroy()
        self._world.reset()

        # Wait and verify ego vehicle is ready before sensor reset
        ego_vehicle = self.get_ego_vehicle()
        max_retries = 10
        for attempt in range(max_retries):
            try:
                if ego_vehicle is not None and ego_vehicle.is_alive:
                    # Configure vehicle physics for better stability
                    physics_control = self.get_ego_vehicle().get_physics_control()
                    # Increase mass slightly for stability
                    physics_control.mass = 1500.0
                    # Adjust wheels for better grip
                    for wheel in physics_control.wheels:
                        wheel.tire_friction = 3.5
                    self.get_ego_vehicle().apply_physics_control(physics_control)
                    # Test ego vehicle validity
                    _ = ego_vehicle.get_location()
                    break
                else:
                    # Wait a bit for ego to be properly spawned
                    time.sleep(0.1)
                    ego_vehicle = self.get_ego_vehicle()
            except:
                if attempt == max_retries - 1:
                    log.error(
                        f"[CARLA] Failed to get valid ego vehicle after {max_retries} attempts"
                    )
                    ego_vehicle = None
                    break
                time.sleep(0.1)
                ego_vehicle = self.get_ego_vehicle()

        self._observer.reset(ego_vehicle)
        self._time_step = 0

        log.info("[CARLA] Environment reset")
        self.obs, _ = self._observer.get_observation(self.get_state())
        # for key in self._image_keys:
        #     if key in self.obs:
        #         self.obs[key] = self.obs[key].astype(np.float32) / 255.0
        self.prev_action = None
        self.current_action = None
        self.prev_steer = 0.0
        self.prev_acc = 0.0
        # Clear stale trackers from previous episode
        if hasattr(self, "prev_wpt_dist"):
            del self.prev_wpt_dist
        if hasattr(self, "prev_steer_diff"):
            del self.prev_steer_diff

        if self._monitor and hasattr(self._monitor, "reset"):
            self._monitor.reset()

        return self.obs

    def close(self):
        """Clean up the environment and release CARLA."""
        try:
            if hasattr(self, "_world") and self._world is not None:
                self._world._set_synchronous_mode(False)
        except Exception:
            pass
        try:
            if hasattr(self, "_observer") and self._observer is not None:
                self._observer.destroy()
        except Exception:
            pass

    def get_vehicle_control(self, action):
        """
        Convert actions in the action space to vehicle control in CARLA
        Includes steering smoothing for jerk-free driving and minimum speed enforcement.
        """
        action_config = self._config.action

        # Correctly decode action (One-hot vector to integer index)
        if isinstance(action, np.ndarray) and action.ndim > 0:
            if action.size > 1:
                action_idx = np.argmax(action)
            else:
                action_idx = int(action)
        else:
            action_idx = int(action)

        # Calculate acceleration and steering
        if action_config.discrete:
            n_steer = len(action_config.discrete_steer)
            n_acc = len(action_config.discrete_acc)
            acc_idx = np.clip(int(action_idx // n_steer), 0, n_acc - 1)
            steer_idx = np.clip(int(action_idx % n_steer), 0, n_steer - 1)
            acc = action_config.discrete_acc[acc_idx]
            steer = action_config.discrete_steer[steer_idx]
            self.n_steer = n_steer  # for r_smoothness calculation
        else:
            acc = action[0]
            steer = action[1]

        # ============================================================
        # STEERING SMOOTHING - Prevent jerky steering
        # ============================================================
        target_steer = steer
        alpha_steer = 0.3  # Match baseline DreamerV3 smoothing factor
        steer = self.prev_steer * (1 - alpha_steer) + target_steer * alpha_steer

        # Limit steering change rate (slew rate) to stabilize physics
        max_steer_delta = 0.1 
        steer_delta = steer - self.prev_steer
        if steer_delta > max_steer_delta:
            steer = self.prev_steer + max_steer_delta
        elif steer_delta < -max_steer_delta:
            steer = self.prev_steer - max_steer_delta
        
        self.prev_steer = steer

        # Convert acceleration to throttle and brake
        if acc > 0:
            throttle = np.clip(acc / 3.0, 0, 1)
            brake = 0
        else:
            throttle = 0
            brake = np.clip(-acc / 3.0, 0, 1)

        return carla.VehicleControl(
            throttle=float(throttle),
            steer=float(-steer),
            brake=float(brake),
            hand_brake=False,
            manual_gear_shift=False,
        )

    def _is_terminal(self):
        terminal_conds = self.get_terminal_conditions()
        terminal = False
        for k, v in terminal_conds.items():
            if v:
                log.info(f"[CARLA] Terminal condition triggered: {k}")
                terminal = True
            terminal_conds[k] = np.array([v], dtype=np.bool_)
        if terminal:
            terminal_conds["episode_timesteps"] = self._time_step
        terminal_conds["terminal"] = terminal
        return terminal, terminal_conds

    def step(self, action):
        self.prev_action = self.current_action
        self.current_action = action

        # IPC OPTIMIZATION: Batch control application and world step
        control = self.get_vehicle_control(action)
        ego_vehicle = self.get_ego_vehicle()

        # Apply control
        ego_vehicle.apply_control(control)

        # IPC OPTIMIZATION: Combine world step with minimal state queries
        self._world.step()
        self._time_step += 1

        # REFRESH: Always get fresh data for zero-lag and high fidelity
        env_state = self.get_state()
        env_state["step_count"] = self._time_step
        reward, reward_info = self.reward()

        # Always get fresh observation to prevent flickering
        self.obs, obs_info = self._observer.get_observation(env_state)

        # Always check terminal conditions for safety
        is_terminal, terminal_conds = self._is_terminal()

        info = {
            **env_state,
            **terminal_conds,
            **obs_info,
            **reward_info,
            "action": action,
            "reward": reward,  # HUD needs this for left panel reward breakdown
        }
        if self._config.eval:
            info = {f"eval_{k}": v for k, v in info.items()}
            self.obs = {**self.obs, **info}
        if self._config.display.enable:
            self._render(self.obs, info)

        self.last_info = info
        return (self.obs, reward, is_terminal, info)

    def is_collision(self):
        """
        Check if the ego vehicle is in collision.
        You must include 'collsion' in observation.names to use this method.
        """
        return self.obs["collision"][0] > 0

    def _render(self, obs, info):
        if self._monitor:
            self._monitor.render(obs, info)
