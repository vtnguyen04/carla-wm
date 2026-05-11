import torch
import numpy as np
import os
import sys
import pathlib
import matplotlib.pyplot as plt
import cv2
import ruamel.yaml as yaml
from sklearn.decomposition import PCA

# Add project roots
sys.path.append(os.getcwd())
sys.path.append('vjepa_project')

from torch_wm.models.wm_agent import WMAgent
from torch_wm.structs import AttrDict
from torch_wm.utils.preprocessing import preprocess_obs
import embodied

def main():
    print("--- 640x360 FULL PIPELINE TEST ---")
    
    device = "cpu" # FORCE CPU
    print(f"Using device: {device}")
    
    # 1. Load Config
    config_path = "torch_wm/rl/config/vjepa_twister.yaml"
    with open(config_path, 'r') as f:
        model_configs = yaml.YAML(typ="safe").load(f)
    config = AttrDict(model_configs["defaults"])
    
    # Override for 640x360
    config.batch_size = 1
    config.batch_length = 2
    config.image_size = [360, 640] 
    config.tubelet_size = 2 
    
    # 2. Initialize Agent
    print("Initializing WMAgent...")
    config.trainable = False 
    # Force disable heads to ensure minimal footprint
    config.modules = {"losses": {"kl": {"enabled": False}, "reward": {"enabled": False}}}
    
    agent = WMAgent(env_name="test_vjepa", override_config=config, skip_env=True)
    agent.to(device)
    agent.eval()
    
    # 3. Load Real Expert Data
    replay_dir = pathlib.Path("./expert_data/replay")
    npz_files = sorted(list(replay_dir.glob("*.npz")), key=lambda x: x.stat().st_mtime, reverse=True)
    if not npz_files:
        print("Error: No data found.")
        return
        
    print(f"Loading data: {npz_files[0].name}")
    data = np.load(npz_files[0])
    
    # Get T=2 frames from middle
    total_frames = data["camera"].shape[0]
    mid_idx = total_frames // 2
    obs_raw = data["camera"][mid_idx:mid_idx+2] 
    
    # 4. Forward Pass
    print("Running Inference...")
    s = {"camera": torch.from_numpy(obs_raw).permute(0, 3, 1, 2).float().unsqueeze(0).to(device) / 255.0}
    a = torch.zeros((1, 2, 15)).to(device)
    r = torch.zeros((1, 2)).to(device)
    d = torch.zeros((1, 2)).to(device)
    f = torch.ones((1, 2)).to(device)

    with torch.no_grad():
        losses = agent.world_model((s, a, r, d, f))
    
    print("\n--- FORWARD SUCCESS ---")
    
    # 5. Visualize Latent PCA
    print("\nGenerating Latent PCA visualization...")
    with torch.no_grad():
        s_proc = preprocess_obs(s, device=device, precision="32")
        vit_input = s_proc["camera"].permute(0, 2, 1, 3, 4) 
        vit_input = (vit_input - agent.encoder_network.mean.to(device)) / agent.encoder_network.std.to(device)
        tokens = agent.encoder_network.model(vit_input)
        
    tokens_np = tokens.squeeze(0).cpu().numpy()
    pca = PCA(n_components=3)
    pca_feat = pca.fit_transform(tokens_np)
    
    for i in range(3):
        pca_feat[:, i] = (pca_feat[:, i] - pca_feat[:, i].min()) / (pca_feat[:, i].max() - pca_feat[:, i].min())
        
    # Spatial: 640/16=40, 360/16=22.5 -> 22.
    pca_spatial = pca_feat.reshape(1, 22, 40, 3)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(obs_raw[0])
    axes[0].set_title(f"Expert Frame (640x360)")
    axes[0].axis('off')
    
    pca_res = cv2.resize(pca_spatial[0], (640, 360), interpolation=cv2.INTER_NEAREST)
    axes[1].imshow(pca_res)
    axes[1].set_title(f"V-JEPA Latent PCA (40x22 patches)")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig("vjepa_wm_final_640x360.png", dpi=150)
    print("--- SUCCESS! Saved to vjepa_wm_final_640x360.png ---")

if __name__ == "__main__":
    main()
