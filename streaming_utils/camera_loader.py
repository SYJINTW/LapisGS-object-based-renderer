import scene.dataset_readers as dataset_readers
import numpy as np
import math
import torch
from torch import nn
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2
from PIL import Image

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        resized_image_rgb = PILtoTorch(image, resolution)
        
        # 1. 先把 RGB 和 Alpha 分開拿出來
        gt_rgb = resized_image_rgb[:3, ...]
        self.alpha_mask = None
        if resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
            
            # 【關鍵修改】執行 Premultiply Alpha
            # 將 RGB 乘上 Alpha，這會去除邊緣混合到的白色雜訊
            gt_rgb = gt_rgb.to(self.data_device)
            gt_image = gt_rgb * self.alpha_mask
            
        else: 
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))
            gt_image = gt_rgb.to(self.data_device) # 如果沒有 alpha，就直接用 RGB
            
        # 處理測試集的遮罩邏輯 (保持原本邏輯不變)
        if train_test_exp and is_test_view:
            if is_test_dataset:
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0
        
        # 儲存最終圖像
        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        # <<<< [YC] Change to new ground truth image loading with Premultiplied Alpha
        
        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            self.depth_mask = torch.ones_like(self.alpha_mask)
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                if depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

def load_camera_from_eyenavgs_config(camera, 
                                    width=800, height=800,
                                    data_device="cuda"):
    """
    camera: dict containing camera parameters, including:
    - "transform_matrix": 4x4 list representing the camera-to-world transformation matrix
    - "camera_angle_x": horizontal field of view in radians
    """
    idx = 0

    position = camera["position"]
    rotation_matrix = camera["rotation"]
    
    R = np.transpose(rotation_matrix)
    T = np.array([position[0], position[1], position[2]])
    
    # # NeRF 'transform_matrix' is a camera-to-world transform
    # c2w = np.array(camera["transform_matrix"])
    # # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
    # c2w[:3, 1:3] *= -1

    # # get the world-to-camera transform and set R, T
    # w2c = np.linalg.inv(c2w)
    # R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
    # T = w2c[:3, 3]
    
    FovX = focal2fov(fov2focal(camera["camera_angle_x"], width), width)
    FovY = focal2fov(fov2focal(camera["camera_angle_y"], height), height)
    
    # fovx = camera["camera_angle_x"]
    # if camera.get("camera_angle_y") is not None:
    #     fovy = camera["camera_angle_y"]
    # else:
    #     fovy = focal2fov(fov2focal(fovx, width), height)
    # FovY = fovy 
    # FovX = fovx
    
    cam_info = dataset_readers.CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path="None", image_name="None",
                            width=width, height=height, depth_path="None", 
                            depth_params=None, is_test=False)
    
    image = Image.open("/home/syjintw/Desktop/NUS/dataset/my_testing_dataset/longdress/1051/test/r_0.png")
    resolution = round(width/1.0), round(height/1.0)
    
    cam = Camera(resolution, colmap_id=cam_info.uid, 
                R=cam_info.R, T=cam_info.T, 
                FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                depth_params=cam_info.depth_params,
                image=image, invdepthmap=None,
                image_name=cam_info.image_name, uid=cam_info.uid, 
                data_device=data_device,
                train_test_exp=None, is_test_dataset=None, 
                is_test_view=cam_info.is_test)
    
    return cam

def load_camera_from_streaming_config(camera, 
                                    width=800, height=800,
                                    data_device="cuda"):
    """
    camera: dict containing camera parameters, including:
    - "transform_matrix": 4x4 list representing the camera-to-world transformation matrix
    - "camera_angle_x": horizontal field of view in radians
    """
    idx = 0

    # NeRF 'transform_matrix' is a camera-to-world transform
    c2w = np.array(camera["transform_matrix"])
    # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
    c2w[:3, 1:3] *= -1

    # get the world-to-camera transform and set R, T
    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]
    
    fovx = camera["camera_angle_x"]
    if camera.get("camera_angle_y") is not None:
        fovy = camera["camera_angle_y"]
    else:
        fovy = focal2fov(fov2focal(fovx, width), height)
    FovY = fovy 
    FovX = fovx
    
    cam_info = dataset_readers.CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path="None", image_name="None",
                            width=width, height=height, depth_path="None", 
                            depth_params=None, is_test=False)
    
    image = Image.open("/home/syjintw/Desktop/NUS/dataset/my_testing_dataset/longdress/1051/test/r_0.png")
    resolution = round(width/1.0), round(height/1.0)
    
    cam = Camera(resolution, colmap_id=cam_info.uid, 
                R=cam_info.R, T=cam_info.T, 
                FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                depth_params=cam_info.depth_params,
                image=image, invdepthmap=None,
                image_name=cam_info.image_name, uid=cam_info.uid, 
                data_device=data_device,
                train_test_exp=None, is_test_dataset=None, 
                is_test_view=cam_info.is_test)
    
    return cam
    
def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))