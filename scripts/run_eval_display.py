import torch
import numpy as np
import embodied
import pathlib
import sys
import pygame
import ruamel.yaml as yaml

# Add project root to path
root = pathlib.Path(__file__).parent.parent
sys.path.append(str(root))
from torch_wm.rl._setup_path import setup
setup()

import carla_env
import embodied
from torch_wm.rl.agent import TwisterAgent

def wrap_env(env, config):
    args = config.wrapper
    env = embodied.wrappers.InfoWrapper(env)
    for name, space in env.act_space.items():
        if name == "reset": continue
        elif space.discrete: env = embodied.wrappers.OneHotAction(env, name)
        else: env = embodied.wrappers.NormalizeAction(env, name)
    env = embodied.wrappers.ExpandScalars(env)
    return env

def run_display():
    print("🎮 Đang khởi động màn hình hiển thị Pygame cho Twister Agent...")
    
    # 1. Config
    config_path = pathlib.Path("torch_wm/rl/config/twister.yaml")
    with open(config_path, 'r') as f:
        model_configs = yaml.YAML(typ="safe").load(f)
    config = embodied.Config(model_configs["defaults"])
    
    # 2. Setup Env & Agent
    from embodied.envs import from_gym
    env, env_config = carla_env.create_task("carla_navigation", ["--env.world.carla_port", "2000"])
    env = from_gym.FromGym(env)
    env = wrap_env(env, config) # Apply wrappers here
    
    step = embodied.Counter()
    # Chạy model trên CPU để dành GPU cho việc Render CARLA + Pygame (tránh OOM)
    agent = TwisterAgent(env.obs_space, env.act_space, step, config.update({'device': 'cpu'}))
    
    # 3. Initialize Pygame
    pygame.init()
    width, height = 128, 64 # 2 ảnh 64x64 cạnh nhau
    scale = 8 # Phóng to 8 lần để dễ nhìn (1024x512)
    display = pygame.display.set_surface = pygame.display.set_mode((width * scale, height * scale))
    pygame.display.set_caption("🛰️ WMAgent - CARLA Real-time Display")
    clock = pygame.time.Clock()

    print("🌍 CARLA Connected. Driving started!")
    obs, info = env.step({'action': env.act_space['action'].sample(), 'reset': np.array(True)})
    state = None

    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Agent ra quyết định
            action, state = agent.policy(obs, state, mode="eval")
            
            # Bổ sung khóa reset bắt buộc cho framework embodied
            action['reset'] = np.array(False)
            
            # Môi trường thực thi
            obs, info = env.step(action)
            
            # --- RENDER ---
            # Ghép camera và birdeye
            cam = obs["camera"] # (64, 64, 3)
            bird = obs["birdeye_wpt"] # (64, 64, 3)
            combined = np.hstack([cam, bird]) # (64, 128, 3)
            
            # Chuyển đổi sang Pygame Surface
            surface = pygame.surfarray.make_surface(combined.swapaxes(0, 1))
            surface = pygame.transform.scale(surface, (width * scale, height * scale))
            
            display.blit(surface, (0, 0))
            pygame.display.flip()
            
            # Giới hạn 10 FPS (khớp với CARLA)
            clock.tick(10)
            
            if obs["is_last"]:
                print("🏁 Episode finished. Resetting...")
                obs, info = env.step({'action': env.act_space['action'].sample(), 'reset': np.array(True)})
                state = None

    except KeyboardInterrupt:
        print("🛑 Stopped by user.")
    finally:
        pygame.quit()
        env.close()

if __name__ == "__main__":
    run_display()
