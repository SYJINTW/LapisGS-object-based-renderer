import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

def load_with_opencv(path):
    # Load as grayscale (0) or color (1)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
    
    if img is None:
        raise FileNotFoundError(f"Image not found at {path}")

    # Convert to float32 and normalize to 0-1
    img_normalized = img.astype(np.float32) / 255.0
    
    # Convert to Tensor and add Channel dimension: (H, W) -> (1, H, W)
    tensor_img = torch.from_numpy(img_normalized).unsqueeze(0)
    
    return tensor_img

def analyze_distortion(img1, img2, output_filename="distortion_map.png"):
    """
    Calculates distortion between two tensors, prints stats, and saves a heatmap.
    Args:
        img1, img2: Tensors of shape (C, H, W) or (1, H, W) with values 0-1.
    """
    # 1. Calculate Absolute Difference (Distortion)
    # If RGB, we often average across channels to get a single 2D map: (H, W)
    # If Grayscale (1, H, W), .mean(dim=0) just squeezes the channel.
    distortion = torch.abs(img1 - img2)
    distortion_map = distortion.mean(dim=0) 

    # 2. Analysis
    max_dist = distortion_map.max().item()
    min_dist = distortion_map.min().item()
    mean_dist = distortion_map.mean().item()
    
    print(f"--- Distortion Analysis ---")
    print(f"Max Distortion:  {max_dist:.6f}")
    print(f"Min Distortion:  {min_dist:.6f}")
    print(f"Mean Distortion: {mean_dist:.6f}")

    # 3. Store as Image (Heatmap)
    # We use Matplotlib to apply a colormap (e.g., 'inferno') which makes 
    # small errors easier to see than standard grayscale.
    plt.figure(figsize=(10, 8))
    
    # Move to CPU and Numpy for plotting
    data = distortion_map.cpu().numpy()
    
    # 'inferno' or 'viridis' are good for heatmaps. 'gray' for black & white.
    # vmin=0, vmax=1 ensures the scale is absolute (0 to 100% error)
    plt.imshow(data, cmap='inferno', vmin=0, vmax=max_dist) 
    plt.colorbar(label='Absolute Error (0-1)')
    plt.title(f"Distortion Map (Max: {max_dist:.4f})")
    
    # Save without whitespace
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"Distortion map saved to: {output_filename}")

# --- Usage ---
# image_name = "00005"
# path1 = f"/mnt/data1/syjintw/NUS/gaussian-splatting/output/materials/train/ours_30000/renders/{image_name}.png"
# path2 = f"/mnt/data1/syjintw/NUS/gaussian-splatting/output/materials/train/ours_30000_merge/renders/{image_name}.png"

# image_name = "00005"
# path1 = f"/mnt/data1/syjintw/NUS/gaussian-splatting/output/materials/train/ours_30000/renders/{image_name}.png"
# path2 = f"/mnt/data1/syjintw/NUS/gaussian-splatting/output/materials/train/ours_30000_original/renders/{image_name}.png"

image_name = "00002"
path1 = f"/mnt/data1/syjintw/NUS/gaussian-splatting/output/materials/train/ours_30000_gs_dw2/renders/{image_name}.png"
path2 = f"/mnt/data1/syjintw/NUS/gaussian-splatting/output/materials/train/ours_30000_img_dw2/renders/{image_name}.png"


# Load
img1 = load_with_opencv(path1)
img2 = load_with_opencv(path2)

# Ensure shapes match before analysis
if img1.shape != img2.shape:
    print(f"Warning: Shapes differ! {img1.shape} vs {img2.shape}")
    # Optional: Resize img2 to match img1
    # img2 = transforms.Resize((img1.shape[1], img1.shape[2]))(img2)

analyze_distortion(img1, img2)