import torch
import numpy as np
import os
import subprocess
from decord import VideoReader
import sys
import matplotlib.pyplot as plt
import cv2
import argparse

# Add vjepa_project to sys.path to import modules
sys.path.append('vjepa_project')

# Import the correct V-JEPA 2.1 model architecture
from app.vjepa_2_1.models.vision_transformer import vit_large
import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from sklearn.decomposition import PCA

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
PT_MODEL_PATH = "models/vjepa2_1_vitl_dist_vitG_384.pt"

def load_pretrained_vjepa_pt_weights(model, pretrained_weights):
    checkpoint = torch.load(pretrained_weights, weights_only=True, map_location="cpu")
    # V-JEPA 2.1 vit_large uses 'ema_encoder'
    pretrained_dict = checkpoint["ema_encoder"]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    msg = model.load_state_dict(pretrained_dict, strict=True)
    print(f"Pretrained weights loaded with msg: {msg}")

def build_pt_video_transform(img_size):
    short_side_size = int(256.0 / 224 * img_size)
    transform = video_transforms.Compose([
        video_transforms.Resize(short_side_size, interpolation="bilinear"),
        video_transforms.CenterCrop(size=(img_size, img_size)),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])
    return transform

def get_video_and_raw_frames(sample_video_path, img_size=384, max_frames=32):
    if not os.path.exists(sample_video_path):
        video_url = "https://www.w3schools.com/html/mov_bbb.mp4"
        subprocess.run(["wget", video_url, "-O", sample_video_path], check=True)

    vr = VideoReader(sample_video_path)
    frame_indices = np.linspace(0, len(vr) - 1, max_frames, dtype=int)
    video = vr.get_batch(frame_indices).asnumpy() # T, H, W, C
    
    raw_frames = []
    short_side_size = int(256.0 / 224 * img_size)
    for frame in video:
        h, w, _ = frame.shape
        if h < w:
            new_h, new_w = short_side_size, int(w * short_side_size / h)
        else:
            new_h, new_w = int(h * short_side_size / w), short_side_size
        frame = cv2.resize(frame, (new_w, new_h))
        y = (new_h - img_size) // 2
        x = (new_w - img_size) // 2
        frame = frame[y:y+img_size, x:x+img_size]
        raw_frames.append(frame)
    raw_frames = np.stack(raw_frames)
    
    return video, raw_frames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="another_video.mp4", help="Path to input video")
    args = parser.parse_args()

    print("Initializing V-JEPA 2.1 ViT-Large model...")
    # Exact kwargs from V-JEPA 2.1 hubconf
    vit_encoder_kwargs = dict(
        patch_size=16,
        img_size=(384, 384),
        num_frames=32,
        tubelet_size=2,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
    )
    model = vit_large(**vit_encoder_kwargs)
    load_pretrained_vjepa_pt_weights(model, PT_MODEL_PATH)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    video_transform = build_pt_video_transform(img_size=384)
    video_data, raw_frames = get_video_and_raw_frames(args.video, max_frames=32)
    
    video_tensor = torch.from_numpy(video_data).permute(0, 3, 1, 2).float() # T, C, H, W
    x_pt = video_transform(video_tensor).unsqueeze(0).to(device) # 1, C, T, H, W
    
    with torch.inference_mode():
        features = model(x_pt) # 1, num_tokens, embed_dim
        
    features_flat = features.squeeze(0).cpu().numpy()
    
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(features_flat)
    
    # Normalize PCA features to [0, 1] range to view as RGB.
    for i in range(3):
        min_val = pca_features[:, i].min()
        max_val = pca_features[:, i].max()
        pca_features[:, i] = (pca_features[:, i] - min_val) / (max_val - min_val)
        
    T_patches, H_patches, W_patches = 16, 24, 24
    pca_spatial = pca_features.reshape(T_patches, H_patches, W_patches, 3)
    
    t_indices = [0, 5, 10, 15]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i, t_idx in enumerate(t_indices):
        raw_frame = raw_frames[t_idx * 2]
        axes[0, i].imshow(raw_frame)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original Video')
            
        pca_img = pca_spatial[t_idx] # 24x24x3
        pca_img_resized = cv2.resize(pca_img, (384, 384), interpolation=cv2.INTER_NEAREST)
        axes[1, i].imshow(pca_img_resized)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('V-JEPA 2.1 Latent (PCA)')
            
    plt.tight_layout()
    plt.savefig('vjepa_latent_pca.png')
    print("Saved visualization to vjepa_latent_pca.png")

if __name__ == "__main__":
    main()