import pathlib
import sys
import warnings
import torch
import numpy as np
import pygame
import carla
import cv2
import ruamel.yaml as yaml
from tqdm import tqdm
import math

# Thêm root vào path
root = pathlib.Path(__file__).parent.parent
sys.path.append(str(root))
from torch_wm.rl._setup_path import setup
setup()

import carla_env
import embodied
from carla_env.toolkit.monitor.pygame_monitor import EnvMonitorPygame

# IMPORT BỘ ĐIỀU KHIỂN CHUẨN
from torch_wm.modules.controllers import PIDController
from carla_env.toolkit.carla_manager.utils import TTCCalculator

def get_speed(vehicle):
    vel = vehicle.get_velocity()
    return 3.6 * np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

class ManualDriver:
    def __init__(self, d_steer, d_acc):
        self.d_steer, self.d_acc = d_steer, d_acc
        self.manual_steer = 0.0

    def plan(self, keys):
        desired_acc = 0.0

        # Steering Logic
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.manual_steer = min(1.0, self.manual_steer + 0.2) # Dương (+) là bẻ TRÁI
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.manual_steer = max(-1.0, self.manual_steer - 0.2) # Âm (-) là bẻ PHẢI
        else:
            # Auto center
            if self.manual_steer > 0: self.manual_steer = max(0.0, self.manual_steer - 0.2)
            elif self.manual_steer < 0: self.manual_steer = min(0.0, self.manual_steer + 0.2)

        # Acceleration Logic
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            desired_acc = 2.0
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            desired_acc = -5.0
        else:
            desired_acc = 0.0

        acc_idx = np.argmin(np.abs(np.array(self.d_acc) - desired_acc))
        steer_idx = np.argmin(np.abs(np.array(self.d_steer) - self.manual_steer))

        status = "MANUAL - Up/Down/Left/Right"
        return acc_idx * len(self.d_steer) + steer_idx, self.manual_steer, status, 0.0

def wrap_env(env, config):
    args = config.wrapper
    env = embodied.wrappers.InfoWrapper(env)
    if args.get("length", 0): env = embodied.wrappers.TimeLimit(env, args.length, args.get("reset", True))
    for name, space in env.act_space.items():
        if name == "reset": continue
        if space.discrete: env = embodied.wrappers.OneHotAction(env, name)
    env = embodied.wrappers.ExpandScalars(env)
    return env

def main():
    config_path = pathlib.Path("torch_wm/rl/config/twister.yaml")
    with open(config_path, 'r') as f:
        model_configs = yaml.YAML(typ="safe").load(f)
    config = embodied.Config(model_configs["defaults"])

    steer_str = ",".join([str(x) for x in config.env_params.action.discrete_steer])

    # 1. Setup Env (EVAL=FALSE ĐỂ KHÔNG BỊ TRỄ LÁI)
    env, env_config = carla_env.create_task("carla_navigation", [
        "--env.world.carla_port", "2000",
        "--env.params.num_vehicles", "250",
        "--env.display.hires", "True",
        "--env.display.width", "1280",
        "--env.display.height", "720",
        "--env.eval", "False",
        "--env.action.discrete_steer", steer_str
    ])

    monitor = EnvMonitorPygame(env_config.env)
    action_spec = env_config.env.action
    d_steer, d_acc = action_spec.discrete_steer, action_spec.discrete_acc
    total_actions = len(d_steer) * len(d_acc)

    from embodied.envs import from_gym
    env = wrap_env(from_gym.FromGym(env), config)
    env = embodied.BatchEnv([env], parallel=False)

    replay = embodied.replay.Uniform(32, 1000000, pathlib.Path("./expert_data/replay"))

    print(f"\n🚀 FINAL MANUAL DRIVER ONLINE. (Action Space: {total_actions})")
    print("Mẹo: Dùng [W/A/S/D] hoặc [Mũi tên] để điều khiển xe trên cửa sổ Pygame. Đừng nhấp ra ngoài để không mất Focus!")

    total_steps = 0
    try:
        for ep in range(1, 11):
            print(f"🌟 EP {ep} START")
            obs = env.step({'action': np.zeros((1, total_actions)), 'reset': np.array([True])})
            raw_env = env._envs[0]._env
            expert = ManualDriver(d_steer, d_acc)
            ep_reward, done, sim_time = 0.0, False, 0.0
            monitor.reset()

            while not done:
                if not pygame.get_init(): break
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: raise KeyboardInterrupt

                ego = raw_env.get_ego_vehicle()
                wpts, _ = raw_env.ego_planner.run_step()
                sim_time += 0.05

                # Plan using Pygame Keys
                keys = pygame.key.get_pressed()
                action_idx, v_steer, status, lead_d = expert.plan(keys)
                one_hot = np.zeros(total_actions, dtype=np.float32)
                one_hot[action_idx] = 1.0

                # --- [ELASTIC AR WAYPOINTS] ---
                # Chiều dài 15 mét theo ý user.
                # Lược bớt điểm để vẽ nét mượt mà dần (cách nhau 1.0m).
                if len(wpts) > 1:
                    max_draw_dist = min(15.0, max(0.1, lead_d - 5.0))
                    truncated_wpts = [wpts[0]]
                    acc_dist = 0.0
                    last_drawn_pt = np.array(wpts[0][:2])
                    for i in range(1, len(wpts)):
                        pt = np.array(wpts[i][:2])
                        step_dist = np.linalg.norm(pt - np.array(wpts[i-1][:2]))
                        acc_dist += step_dist

                        if np.linalg.norm(pt - last_drawn_pt) >= 1.0:
                            truncated_wpts.append(wpts[i])
                            last_drawn_pt = pt

                        if acc_dist >= max_draw_dist:
                            break
                    wpts = truncated_wpts

                # Step
                obs, info = env.step({'action': np.expand_dims(one_hot, 0), 'reset': np.array([False])})
                ep_reward += obs['reward'][0]
                done = obs['is_last'][0]

                # --- [A-Z AUDIT] DEBUG RESOLUTION ---
                if total_steps % 100 == 0:
                    print(f"🔎 DEBUG INFO KEYS: {list(info.keys())}")
                    if 'camera_display' in info:
                         print(f"📷 [HIRES CHECK] info['camera_display'] Shape: {info['camera_display'].shape}")

                # --- LẤY DISPLAY HIRES ---
                monitor_obs = {k: v[0] for k, v in obs.items()}
                for handler in raw_env._observer._display_handlers:
                    if hasattr(handler, "_hires_data") and handler._hires_data is not None:
                        monitor_obs[handler._config.key] = handler._hires_data
                    else:
                        hd_obs, _ = handler.get_observation(raw_env.get_state())
                        if handler._config.key in hd_obs:
                            monitor_obs[handler._config.key] = hd_obs[handler._config.key]

                monitor.sim_time = sim_time

                # Fetch advanced metrics that were bypassed by embodied BatchEnv
                ttc, dist_front = TTCCalculator.get_ttc_and_distance(ego, raw_env._world.carla_world, raw_env._world.carla_map)
                ego_pos = np.array([ego.get_location().x, ego.get_location().y])
                wpt_dist = np.linalg.norm(np.array(wpts[0][:2]) - ego_pos) if len(wpts) > 0 else 0.0

                hud_info = {
                    "waypoints": wpts, "ego_transform": ego.get_transform(),
                    "throttle": ego.get_control().throttle, "steer": ego.get_control().steer, "brake": ego.get_control().brake,
                    "speed_norm": get_speed(ego) / 3.6,
                    "reward": float(obs["reward"][0]), # Pass instantaneous reward!
                    "wpt_dist": wpt_dist,
                    "location": [ego.get_location().x, ego.get_location().y, ego.get_location().z],
                    "town": "Town01", "is_last": done, "collision": obs["collision"][0] > 0,
                    "ttc": ttc, "dist_to_front": dist_front,
                    "sum_travel_distance": getattr(raw_env, "sum_travel_distance", 0.0),
                    "success_dist": env_config.env.get("success_dist", 100.0),
                    "timesteps": raw_env._time_step,
                    "time_limit": env_config.env.terminal.get("time_limit", 1000)
                }
                monitor.render(monitor_obs, hud_info)

                # Save
                replay_step = {k: v[0] for k, v in obs.items() if k in env.obs_space or k in ['reward', 'is_first', 'is_last']}
                replay_step['action'] = one_hot
                replay.add(replay_step)

                total_steps += 1
                if done: break

    except KeyboardInterrupt:
        print("\n🛑 STOP.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ ERROR: {e}")
    finally:
        print("Cleaning up...")
        try: replay.save()
        except: pass
        try: monitor.close()
        except: pass
        try: env.close()
        except: pass
        try: pygame.quit()
        except: pass
        import os, signal
        os.kill(os.getpid(), signal.SIGTERM)

if __name__ == "__main__":
    main()
