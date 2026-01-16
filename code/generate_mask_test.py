import cv2
import numpy as np
import os
import glob

from pathlib import Path

def generate_alpha_mask(
    input_path, output_path
):
    """
    Generates a binary mask from the alpha channel of an image.
    Pixels that are NOT fully transparent (Alpha > 0) become White (255).
    Fully transparent pixels (Alpha == 0) become Black (0).
    """
    
    # 1. Load image with IMREAD_UNCHANGED to keep the Alpha channel
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        print(f"Error: Could not load image at {input_path}")
        return

    # 2. Check if image actually has an Alpha channel (4 channels)
    if image.ndim == 3 and image.shape[2] == 4:
        # Extract the Alpha channel (index 3)
        alpha_channel = image[:, :, 3]
        
        # 3. Create Binary Mask
        # Condition: Wherever alpha is NOT 0, set mask to 255.
        _, mask = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)
        
        # 4. Save the Mask
        output_path = os.path.join(output_path)
        
        cv2.imwrite(output_path, mask)
        print(f"Success: Mask saved to {output_path}")
        
    else:
        print(f"Error: Image '{input_path}' does not have an Alpha channel (Transparency).")

# --- Usage Example ---

# Define paths
scene_name = "lego"
# distances = [1, 2, 3, 4, 5, 6, 7, 8]
# distances = [1, 2, 4, 8]
distances = [3, 5, 6, 7, 9]
# num_of_images = 50
num_of_images = 50

for distance in distances:
    for i in range(num_of_images):
        input_depth_path = f"/home/syjintw/Desktop/NUS/tmp/{scene_name}/results_test_{distance}x/r_{i}.png"  # Change this to your file
        target_dir = Path(f"/home/syjintw/Desktop/NUS/tmp/{scene_name}/results_test_{distance}x/mask")
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Run (Example: Keep everything closer than 1.5 meters/1500 units)
        # Ensure your input path exists before running
        target_path = target_dir / f"r_{i}_mask.png"
        generate_alpha_mask(input_depth_path, target_path)