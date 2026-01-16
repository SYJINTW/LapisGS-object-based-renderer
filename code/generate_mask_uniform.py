import cv2
import numpy as np
import os
import glob

from pathlib import Path

def generate_mask(black_path, white_path, output_path, tolerance=2):
    """
    Generates a binary mask from two rendered images (one with black bg, one with white bg).
    
    Args:
        black_path (str or Path): Path to image rendered on black background.
        white_path (str or Path): Path to image rendered on white background.
        output_path (str or Path): Path to save the output mask.
        tolerance (int): Allow slight deviations from pure black/white (useful for compression artifacts).
                         Set to 0 for strict checking.
    """
    # Load images
    # Ensure paths are strings for cv2
    img_black = cv2.imread(str(black_path))
    img_white = cv2.imread(str(white_path))

    if img_black is None or img_white is None:
        print(f"Error loading images: {black_path} or {white_path}")
        return

    # 1. Identify Background Pixels
    # Condition: Pixel is Black (0,0,0) in first image AND White (255,255,255) in second image
    # We use a small tolerance to handle potential anti-aliasing or slight rendering noise
    
    # Check if pixels in black image are close to [0,0,0]
    is_black_bg = np.all(img_black <= tolerance, axis=2)
    
    # Check if pixels in white image are close to [255,255,255]
    is_white_bg = np.all(img_white >= (255 - tolerance), axis=2)

    # Combine conditions: True if it is background
    is_background = np.logical_and(is_black_bg, is_white_bg)

    # 2. Create Mask
    # Initialize a black mask (all zeros)
    mask = np.zeros_like(is_background, dtype=np.uint8)
    
    # Set Foreground (NOT background) to 255 (White)
    # The user asked to "mask" the background pixels, implying they should be 0 (ignored),
    # and the valid object pixels should be 255.
    mask[~is_background] = 255

    # 3. Save
    cv2.imwrite(str(output_path), mask)

if __name__ == "__main__":
    # --- Usage Example ---
    # Define paths
    scene_name = "lego"
    distances = [1]
    # num_of_images = 50
    num_of_images = 50

    for distance in distances:
        for i in range(num_of_images):
            input_black_path = f"/home/syjintw/Desktop/NUS/lapisgs-output/{scene_name}/opacity/{scene_name}_res1/ours_uniform_{distance}x/ours_30000/renders/{i:05d}.png"  # Change this to your file
            input_white_path = f"/home/syjintw/Desktop/NUS/lapisgs-output/{scene_name}/opacity/{scene_name}_res1/ours_uniform_{distance}x_white/ours_30000/renders/{i:05d}.png"  # Change this to your file
            target_dir = Path(f"/home/syjintw/Desktop/NUS/tmp/{scene_name}/uniform_{distance}x/mask")
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Ensure your input path exists before running
            target_path = target_dir / f"r_{i}_mask.png"
            generate_mask(input_black_path, input_white_path, target_path)