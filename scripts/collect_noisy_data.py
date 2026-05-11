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

class ModularExpert:
    def __init__(self, d_steer, d_acc, target_speed_ms):
        self.d_steer, self.d_acc = d_steer, d_acc
        self.target_speed_ms = target_speed_ms
        # BỘ PID ĐÃ ĐƯỢC TUNE LẠI ĐỂ TRIỆT TIÊU ZIGZAG (Hạ Kp để đỡ gắt, tăng Kd để ghìm xe)
        self.lateral = PIDController(Kp=0.8, Ki=0.02, Kd=0.4)
        self.prev_steer, self.alpha = 0.0, 0.7 # PID phản ứng chuẩn, giữ alpha nhanh

    def plan(self, ego, world, wpts, current_speed_ms, env_info):
        ego_tf = ego.get_transform()
        if len(wpts) > 1:
            # Dynamic Pure Pursuit Lookahead
            ego_pos = np.array([ego_tf.location.x, ego_tf.location.y])
            Ld = max(6.0, current_speed_ms * 1.5) # Look ahead at least 6m
            target_idx = min(len(wpts)-1, 5)
            for i in range(1, len(wpts)):
                d = np.linalg.norm(np.array(wpts[i][:2]) - ego_pos)
                if d >= Ld:
                    target_idx = i
                    break
            else:
                target_idx = len(wpts) - 1

            target_loc = carla.Location(float(wpts[target_idx][0]), float(wpts[target_idx][1]), 0)
            local_target = ego_tf.inverse_transform(target_loc)

            # Calculate heading error to target instead of raw Y to prevent wide zigzag
            alpha_heading_error = math.atan2(local_target.y, local_target.x)

            # --- KHÔI PHỤC DẤU ÂM (-) MỚI LÀ ĐÚNG HƯỚNG ---
            raw_steer = -self.lateral.run(alpha_heading_error)
        else: raw_steer = 0.0

        # Smooth
        smooth_steer = (1 - self.alpha) * self.prev_steer + self.alpha * raw_steer
        self.prev_steer = smooth_steer

        # Radar (Trajectory-aware detection via wpts)
        ttc, lead_dist = TTCCalculator.get_ttc_and_distance(ego, world, world.get_map(), wpts)
        if lead_dist < 0: lead_dist = 200.0 # Clear

        is_red_light = env_info.get("is_red_light", False)
        dist_to_red_light = env_info.get("dist_to_red_light", -1.0)

        # Speed limit & Dynamic Curvature Adaptation
        speed_limit_ms = ego.get_speed_limit() / 3.6
        base_speed_ms = min(speed_limit_ms * 0.9, self.target_speed_ms)

        # Look-ahead up to 15 waypoints (~15 meters) to detect turns
        max_yaw_diff = 0.0
        ego_yaw = ego_tf.rotation.yaw
        if wpts is not None and len(wpts) > 0:
            for wp in wpts[:15]:
                # Normalize yaw diff to [-180, 180]
                yaw_diff = (wp[2] - ego_yaw + 180) % 360 - 180
                if abs(yaw_diff) > max_yaw_diff:
                    max_yaw_diff = abs(yaw_diff)

        # Dynamically suppress target speed based on curve severity
        if max_yaw_diff > 10.0:
            curve_penalty = 1.0 - min(max_yaw_diff / 90.0, 0.6) # Reduces up to 60% of speed on a 90-deg turn
            target_speed = max(base_speed_ms * curve_penalty, 3.5) # Minimum cornering speed 3.5m/s (12.6 km/h)
        else:
            target_speed = base_speed_ms

        if lead_dist < 7.0 or (0.0 < ttc < 2.0):  # Priority 1: Instant Stop Zone
            desired_acc, status = -3.0, "EMERGENCY"
        elif (0.0 < ttc < 3.0) or lead_dist < 12.0:
            # Priority 2: Active hazard (crash in < 3s OR entering 12m warning zone)
            if current_speed_ms > 0.1:
                desired_acc, status = -2.0, f"BRAKING {ttc:.1f}s"
            else:
                desired_acc, status = -3.0, "HOLD"
        elif is_red_light and dist_to_red_light > 0:
            # Priority 3: Red Light Approach
            if dist_to_red_light < 6.0 or (dist_to_red_light < 15.0 and current_speed_ms < 0.5):
                # 0m-6.0m OR almost stopped: Firm lock to prevent automatic idle torque creeping
                desired_acc, status = -3.0, "RED STOP"
            elif dist_to_red_light < 10.6:
                # 6.0m-10.6m: Gentle braking (4.6m stopping distance from 20km/h @ -1.0 brake)
                desired_acc, status = -1.0, f"RED BRAKING {dist_to_red_light:.1f}m"
            else:
                # >10.6m: Lift off throttle, coast towards intersection naturally
                desired_acc, status = 0.0, f"RED COAST {dist_to_red_light:.1f}m"
        else:
            speed_diff = target_speed - current_speed_ms
            if speed_diff > 2.0:
                desired_acc, status = 3.0, "HARD ACCEL"
            elif speed_diff > 0.8:
                desired_acc, status = 2.0, "ACCELERATING"
            elif speed_diff > 0.0:
                desired_acc, status = 1.0, "CRUISING"
            elif speed_diff > -0.5:
                desired_acc, status = 0.0, "COASTING"
            else:
                desired_acc, status = -1.0, "LIMIT BRAKE"

        acc_idx = np.argmin(np.abs(np.array(self.d_acc) - desired_acc))
        steer_idx = np.argmin(np.abs(np.array(self.d_steer) - smooth_steer))
        return acc_idx * len(self.d_steer) + steer_idx, smooth_steer, status, lead_dist

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
        "--env.params.num_vehicles", "30",
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

    print(f"\n🚀 FINAL MASTER EXPERT ONLINE. (Action Space: {total_actions})")

    total_steps = 0
    try:
        for ep in range(1, 11):
            print(f"🌟 EP {ep} START")
            obs = env.step({'action': np.zeros((1, total_actions)), 'reset': np.array([True])})

            raw_env = env._envs[0]
            while hasattr(raw_env, '_env'): raw_env = raw_env._env
            if hasattr(raw_env, 'env'): raw_env = raw_env.env # For some nested gym structures
            if hasattr(raw_env, 'unwrapped'): raw_env = raw_env.unwrapped

            expert = ModularExpert(d_steer, d_acc, 20.0/3.6)
            ep_reward, done, sim_time = 0.0, False, 0.0
            drive_mode = "AUTOPILOT"
            explore_frames, explore_action = 0, 0
            is_paused = False
            monitor.reset()

            while not done:
                if not pygame.get_init(): break

                # Check for discrete mode toggle events
                events = getattr(monitor, 'last_events', [])
                if hasattr(monitor, 'last_events'):
                    monitor.last_events = [] # Clear immediately to avoid processing the same key multiple times!

                if len(events) == 0:
                     # Nếu monitor chưa render khung nào (chẳng hạn lúc Pause), tự lấy sự kiện mới
                     events = pygame.event.get()

                for event in events:
                    if event.type == pygame.QUIT: raise KeyboardInterrupt
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_1:
                            drive_mode = "AUTOPILOT"
                            print("🚀 MODE SWITCHED: AUTOPILOT (Expert takes control)")
                        elif event.key == pygame.K_2:
                            drive_mode = "MANUAL"
                            print("🚀 MODE SWITCHED: MANUAL (Human takes control)")
                        elif event.key == pygame.K_p:
                            is_paused = not is_paused
                            print(f"{'⏸️ ĐÃ TẠM DỪNG (Bấm P lần nữa để Đi Tiếp)' if is_paused else '▶️ ĐÃ TIẾP TỤC SIMULATION'}")
                        elif event.key == pygame.K_r or event.key == pygame.K_q:
                            print("🔄 RESET EPISODE KHẨN CẤP...")
                            done = True  # Ép buộc kết thúc vòng lặp để thu thập sang EP tiếp theo

                if is_paused:
                    # Tick the pygame clock so window doesn't freeze, but skip env step
                    monitor.clock.tick(30)
                    pygame.display.flip()
                    continue

                ego = raw_env.get_ego_vehicle()
                wpts, _ = raw_env.ego_planner.run_step()
                sim_time += 0.05
                status = ""

                # --- CONTROLLER SELECTION ---
                env_info = raw_env.last_info if hasattr(raw_env, 'last_info') else {}
                if drive_mode == "AUTOPILOT":
                    expert_idx, v_steer, expert_status, lead_d = expert.plan(ego, raw_env._world.carla_world, wpts, get_speed(ego)/3.6, env_info)

                    # --- NOISY EXPLORATION LOGIC (EPSILON GREEDY WITH ACTION REPEAT) ---
                    # To explore corners (sidewalks, off-center, hard swerves), we can't just explore 1 frame (0.05s).
                    # We must sustain the random action for 5-15 frames to actually go somewhere Out of Distribution!
                    if explore_frames > 0:
                        explore_frames -= 1
                        action_idx = explore_action
                        status = f"RANDOM EXPLORE ({explore_frames} left)"
                    else:
                        # 20% chance to start a new exploration sequence
                        if np.random.rand() < 0.20:
                            explore_frames = np.random.randint(5, 15) # Hold for 0.25s to 0.75s
                            explore_action = np.random.randint(0, total_actions)
                            action_idx = explore_action
                            status = f"RANDOM EXPLORE ({explore_frames} left)"
                        else:
                            action_idx = expert_idx
                            status = expert_status
                else:
                    # MANUAL DRIVING MODE (W, A, S, D)
                    keys = pygame.key.get_pressed()
                    manual_acc = 0.0
                    manual_steer = 0.0
                    if keys[pygame.K_SPACE]:
                        manual_acc = -10.0  # Phanh khẩn cấp
                    elif keys[pygame.K_w]:
                        manual_acc = 3.0
                    elif keys[pygame.K_s]:
                        manual_acc = -5.0

                    # Đảo trục A/D vì carla_base_env.py đang thiết lập: steer = float(-steer)
                    if keys[pygame.K_a]: manual_steer = 0.6    # Sang trái
                    elif keys[pygame.K_d]: manual_steer = -0.6 # Sang phải

                    # Snap to closest discrete action index
                    acc_idx = np.argmin(np.abs(np.array(d_acc) - manual_acc))
                    steer_idx = np.argmin(np.abs(np.array(d_steer) - manual_steer))
                    action_idx = acc_idx * len(d_steer) + steer_idx
                    status = "MANUAL OVERRIDE"
                    if len(wpts) > 0:
                         lead_d = np.linalg.norm(np.array([ego.get_location().x, ego.get_location().y]) - np.array(wpts[0][:2]))
                    else:
                         lead_d = 100.0

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

                env_info = raw_env.last_info if hasattr(raw_env, 'last_info') else {}

                hud_info = {
                    "waypoints": wpts, "ego_transform": ego.get_transform(),
                    "throttle": ego.get_control().throttle, "steer": ego.get_control().steer, "brake": ego.get_control().brake,
                    "speed_norm": get_speed(ego) / 3.6,
                    "reward": float(obs["reward"][0]), # Pass instantaneous reward!
                    "wpt_dist": wpt_dist,
                    "location": [ego.get_location().x, ego.get_location().y, ego.get_location().z],
                    "town": "Town01", "is_last": done, "collision": obs["collision"][0] > 0,
                    "ttc": ttc, "dist_to_front": dist_front,
                    "is_red_light": env_info.get("is_red_light", False),
                    "sum_travel_distance": getattr(raw_env, "sum_travel_distance", 0.0),
                    "success_dist": env_config.env.get("success_dist", 100.0),
                    "timesteps": raw_env._time_step,
                    "time_limit": env_config.env.terminal.get("time_limit", 1000),
                    "drive_mode": drive_mode,
                    **env_info
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
