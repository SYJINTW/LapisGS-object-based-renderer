#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from numbers import Number
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer_lapisgs import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer_lapisgs import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

import torch.nn.functional as F
import json

from streaming_utils.camera_loader import load_camera_from_streaming_config, load_camera_from_eyenavgs_config

class FakePipe:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False

def streaming_render(gs_path_list, gs_res_list, 
                    trace_path,
                    width, height,
                    output_dir, name, 
                    pipeline, background, sh_degree):
    
    render_path = os.path.join(output_dir, name, "renders")
    makedirs(render_path, exist_ok=True)
    
    with open(trace_path, 'r') as f:
        traces = json.load(f)
        
    print("Number of views to render: ", len(traces["frames"]))
    
    # Tricky part: to fit load_camera_from_streaming_config
    camera_angle_x = traces["camera_angle_x"]
    if traces.get("camera_angle_y") is not None:
        print("camera_angle_y found:", traces["camera_angle_y"])
        camera_angle_y = traces["camera_angle_y"]
    else:
        camera_angle_y = None
        
    for frame in traces["frames"]:
        frame["camera_angle_x"] = camera_angle_x
        if camera_angle_y is not None:
            frame["camera_angle_y"] = camera_angle_y
    
    _gs_res = []
    gaussians = GaussianModel(sh_degree)
    for gs_path, gs_res in zip(gs_path_list, gs_res_list):
        obj = GaussianModel(sh_degree)
        obj.load_ply(gs_path)
        gaussians.merge(obj)
        _gs_res.extend([float(gs_res) for _ in range(len(obj.get_xyz))])
    gs_res = torch.tensor(_gs_res, device="cuda")
    
    for idx, _ in enumerate(tqdm([i for i in range(len(traces["frames"]))], desc="Rendering progress")):        
        
        view = load_camera_from_streaming_config(traces["frames"][idx], width=width, height=height)
        # view = load_camera_from_eyenavgs_config(traces["frames"][idx], width=width, height=height)
        bg_color = torch.tensor(background, dtype=torch.float32, device="cuda").view(3, 1, 1)
        bg_color = bg_color.expand(3, view.image_height, view.image_width)
        bg_depth = torch.full((1, view.image_height, view.image_width), 0, dtype=torch.float32, device="cuda")

        # gt = view.original_image[0:3, :, :]
        
        # Only one GaussianModel in gaussians_list for now
        render_results = render(view, gaussians, pipeline,
                                bg_color, bg_depth,
                                gs_res=gs_res)
        
        torchvision.utils.save_image(render_results["render"], os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        
        # if idx >= 500:
        #     break  # [YC] debug: only render one image
         
def render_sets(sh_degree: int = 3,
                gs_path_list : list = [], gs_res_list : list = [],
                trace_path : str = None,
                width : int = 800, height : int = 800,
                name : str = None, 
                output_dir: str = "./results", white_background: bool = False):
    pipeline = FakePipe()
    
    # with open(streaming_config_path, 'r') as f:
    #     streaming_config = json.load(f)
        
    with torch.no_grad():
        bg_color = [1,1,1] if white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        streaming_render(gs_path_list, gs_res_list, 
                        trace_path, 
                        width, height,
                        output_dir, f"{name}",  
                        pipeline, background, sh_degree)
              
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--gs_path_list", nargs='*', type=str, default=[], help="List of paths to additional Gaussian models to render together with the main model.")
    parser.add_argument("--gs_res_list", nargs='*', type=int, default=[], help="List of resolution levels for each additional Gaussian model.")
    parser.add_argument("--name", type=str, help="Name of the streaming scenarios.")
    parser.add_argument("--trace_path", type=str, help="Path to streaming trace JSON file.")
    parser.add_argument("--width", type=int, default=800, help="Width of the rendered images.")
    parser.add_argument("--height", type=int, default=800, help="Height of the rendered images.")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save rendered images.")
    parser.add_argument("--sh_deg", type=int, default=3, help="Spherical harmonics degree.")
    parser.add_argument("--white_bg", action="store_true", help="Use white background for rendering.")
    args = get_combined_args(parser)
    # print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(sh_degree=args.sh_deg, 
                gs_path_list=args.gs_path_list, gs_res_list=args.gs_res_list,
                trace_path=args.trace_path,
                width=args.width, height=args.height,
                name=args.name,
                output_dir=args.output_dir, white_background=args.white_bg
                )
    
    
"""
python render-lapisgs_streaming_trace.py \
--name trace \
--gs_path_list /home/syjintw/Desktop/NUS/dataset/my_testing_gs/bicycle/point_cloud/iteration_30000/point_cloud.ply \
--gs_res_list 1 \
--trace_path /home/syjintw/Desktop/NUS/eyenavgs/user5_bicycle.json \
--sh_deg 3 \
--output_dir /home/syjintw/Desktop/NUS/rendering_results/bicycle \
--width 2064 --height 2272
"""