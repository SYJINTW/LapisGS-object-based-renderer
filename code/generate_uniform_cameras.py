import json
import numpy as np
from pathlib import Path

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def look_at(eye, target, up=np.array([0, 0, 1])):
    """
    Computes the 4x4 camera-to-world transformation matrix.
    Assumes OpenGL style coordinate system (camera looks down -Z).
    """
    z_axis = normalize(eye - target)       # Forward (camera looks towards target, so Z points from target to eye)
    x_axis = normalize(np.cross(up, z_axis)) # Right
    y_axis = normalize(np.cross(z_axis, x_axis)) # Up (orthogonalized)
    
    mat = np.eye(4)
    mat[:3, 0] = x_axis
    mat[:3, 1] = y_axis
    mat[:3, 2] = z_axis
    mat[:3, 3] = eye
    return mat

def calculate_camera_distances(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    distances = []
    
    # print(f"{'Frame':<20} | {'Camera Position (X, Y, Z)':<30} | {'Distance'}")
    # print("-" * 65)

    for frame in data['frames']:
        # Convert list to numpy array
        c2w_matrix = np.array(frame['transform_matrix'])
        
        # Extract the translation vector (Camera Position)
        # This is the first 3 rows of the last column (index 3)
        camera_pos = c2w_matrix[:3, 3]
        
        # Calculate Euclidean distance to origin (0,0,0)
        # distance = sqrt(x^2 + y^2 + z^2)
        dist = np.linalg.norm(camera_pos)
        
        distances.append(dist)
        
        # # Optional: Print details for the first few frames to verify
        # print(f"{frame['file_path']:<20} | {str(np.round(camera_pos, 2)):<30} | {dist:.4f}")

    return distances

def generate_fibonacci_sphere_z_positive(samples, radius):
    """Generates 'samples' points uniformly distributed on a sphere of 'radius'."""
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # Golden angle in radians

    for i in range(samples):
        # 修改這裡：讓 z 從 1 (頂點) 均勻過渡到 0 (赤道)
        z = 1 - (i / float(samples - 1))
        
        radius_at_z = np.sqrt(1 - z * z) # 該高度層的圓半徑
        theta = phi * i
        
        x = np.cos(theta) * radius_at_z
        y = np.sin(theta) * radius_at_z
        
        # 組合座標，確保 z 為正值
        points.append(np.array([x * radius, y * radius, z * radius]))
        
    return points

### 新增：座標轉經緯度函式
def get_lat_lon(x, y, z):
    """
    將 (x, y, z) 轉換為 經度 (Longitude) 與 緯度 (Latitude)
    單位：Degrees (角度)
    Latitude: -90 (南極) ~ 90 (北極), 0 為赤道
    Longitude: -180 ~ 180
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    if r == 0: return 0.0, 0.0

    # 緯度：使用 arcsin(z/r) 計算仰角
    lat_rad = np.arcsin(z / r)
    
    # 經度：使用 arctan2(y, x)
    lon_rad = np.arctan2(y, x)
    
    return np.degrees(lat_rad), np.degrees(lon_rad)

if __name__ == "__main__":
    # Usage
    distance = 1
    
    # Replace 'training_transforms.json' with your actual file name
    training_cam_path = "/home/syjintw/Desktop/NUS/dataset/lego/transforms_train.json"
    output_json = f'/home/syjintw/Desktop/NUS/tmp/lego/uniform_{distance}x/transforms.json'
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)

    num_new_cameras = 50  #! Change this to how many cameras you want

    distances = calculate_camera_distances(training_cam_path)
    distances_mean = np.mean(distances)
    distances_std = np.std(distances)
    print(distances_mean)
    print(distances_std)

    # 2. Generate new positions
    new_positions = generate_fibonacci_sphere_z_positive(num_new_cameras, distances_mean*distance)

    # 3. Create new frames
    new_frames = []
    for i, pos in enumerate(new_positions):
        # Handle the singularity where camera is directly above/below center
        # If camera is at (0,0,Z), we can't use (0,0,1) as up vector.
        if np.abs(pos[0]) < 1e-5 and np.abs(pos[1]) < 1e-5:
            current_up = np.array([0, 1, 0])
        else:
            current_up = np.array([0, 0, 1])

        transform = look_at(pos, np.array([0, 0, 0]), up=current_up)
        
        lat, lon = get_lat_lon(pos[0], pos[1], pos[2])
        
        new_frame = {
            "file_path": f"../results_test_1x/r_{i}", #! fake path
            "rotation": 0.0,  # Standard placeholder
            "transform_matrix": transform.tolist(),
            "latitude": float(f"{lat:.4f}"),  # 格式化保留 4 位小數
            "longitude": float(f"{lon:.4f}")  # 格式化保留 4 位小數
        }
        new_frames.append(new_frame)
        
    # 4. Save to file
    # Load existing data to find average radius
    with open(training_cam_path, 'r') as f:
        data = json.load(f)
        
    output_data = {
        "camera_angle_x": data.get("camera_angle_x", 0.69), # Preserve original FOV
        "frames": new_frames
    }

    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"Successfully generated {num_new_cameras} cameras to '{output_json}'")
    