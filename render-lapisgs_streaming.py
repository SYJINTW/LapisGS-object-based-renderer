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

from streaming_utils.camera_loader import load_camera_from_streaming_config

class FakePipe:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False

def streaming_render(model_path, name, 
                    pipeline, background,
                    streaming_config, sh_degree):
    
    render_path = os.path.join(model_path, name, "renders")
    
    makedirs(render_path, exist_ok=True)
    
    print("Number of views to render: ", len(streaming_config))
    views = [load_camera_from_streaming_config(streaming_config[str(idx)]["camera"]) for idx in range(len(streaming_config))]
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        _gs_res = []
        gaussians = GaussianModel(sh_degree)
        for content in streaming_config[str(idx)]["contents"]:
            obj = GaussianModel(sh_degree)
            obj.load_ply(content["tiles_path"])
            gaussians.merge(obj)
            _gs_res.extend([float(content["resolution"]) for _ in range(len(obj.get_xyz))])
        gs_res = torch.tensor(_gs_res, device="cuda")
        
        streaming_view = load_camera_from_streaming_config(streaming_config[str(idx)]["camera"])
        
        bg_color = torch.tensor(background, dtype=torch.float32, device="cuda").view(3, 1, 1)
        bg_color = bg_color.expand(3, view.image_height, view.image_width)
        bg_depth = torch.full((1, view.image_height, view.image_width), 0, dtype=torch.float32, device="cuda")

        gt = view.original_image[0:3, :, :]
        
        # Only one GaussianModel in gaussians_list for now
        render_results = render(streaming_view, gaussians, pipeline,
                                bg_color, bg_depth,
                                gs_res=gs_res)
        
        torchvision.utils.save_image(render_results["render"], os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        
        # if idx >= 5:
        #     break  # [YC] debug: only render one image
         
def render_sets(sh_degree: int = 3, 
                streaming_config_path : str = None, streaming_scenarios_name : str = None, 
                output_dir: str = "./results", white_background: bool = False):
    pipeline = FakePipe()
    
    with open(streaming_config_path, 'r') as f:
        streaming_config = json.load(f)
        
    with torch.no_grad():
        bg_color = [1,1,1] if white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        streaming_render(output_dir, f"{streaming_scenarios_name}",  
                        pipeline, background,
                        streaming_config, sh_degree)
              
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
    parser.add_argument("--streaming_config_path", type=str, help="Path to streaming configuration JSON file.")
    parser.add_argument("--streaming_scenarios_name", type=str, help="Name of the streaming scenarios.")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save rendered images.")
    parser.add_argument("--sh_deg", type=int, default=3, help="Spherical harmonics degree.")
    parser.add_argument("--white_bg", action="store_true", help="Use white background for rendering.")
    args = get_combined_args(parser)
    # print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(sh_degree=args.sh_deg,
                streaming_config_path=args.streaming_config_path, streaming_scenarios_name=args.streaming_scenarios_name,
                output_dir=args.output_dir, white_background=args.white_bg
                )