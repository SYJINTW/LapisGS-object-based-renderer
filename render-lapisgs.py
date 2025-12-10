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

def render_set(model_path, name, iteration, views, 
               gaussians, gs_res, 
               pipeline, background, train_test_exp, separate_sh):
    
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        #! Change the resolution of rendering images
        # view.image_height = int(view.image_height//2)
        # view.image_width = int(view.image_width//2)
        
        bg_color = torch.tensor(background, dtype=torch.float32, device="cuda").view(3, 1, 1)
        bg_color = bg_color.expand(3, view.image_height, view.image_width)
        bg_depth = torch.full((1, view.image_height, view.image_width), 0, dtype=torch.float32, device="cuda")

        gt = view.original_image[0:3, :, :]
        
        # Only one GaussianModel in gaussians_list for now
        render_results = render(view, gaussians, pipeline,
                        bg_color, bg_depth,
                        gs_res=gs_res,
                        use_trained_exp=train_test_exp, separate_sh=separate_sh)
        
        print("render min/max: ", render_results["render"].min().item(), render_results["render"].max().item()) # [YC] debug
        print("opacity min/max: ", render_results["opacity"].min().item(), render_results["opacity"].max().item()) # [YC] debug

        if args.train_test_exp:
            render_results["render"] = render_results["render"][..., render_results["render"].shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]
        
        # input_tensor = render_results["render"].unsqueeze(0)
        # scaled_tensor = F.interpolate(input_tensor, size=(800, 800), mode='bilinear', align_corners=False)
        # render_results["render"] = scaled_tensor.squeeze(0)
        
        torchvision.utils.save_image(render_results["render"], os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        
        if idx >= 5:
            break  # [YC] debug: only render one image
    
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool, 
                gs_path_list : list = [], 
                gs_res_list : list = [1]):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        # Just use the GaussianModel loaded from dataset
        if len(gs_path_list) == 0:
            if len(gs_res_list) == 0:
                gs_res = torch.tensor([1.0 for _ in range(len(gaussians.get_xyz))], device="cuda")
            elif len(gs_res_list) == 1:
                gs_res = torch.tensor([float(gs_res_list[0]) for _ in range(len(gaussians.get_xyz))], device="cuda")
            else:
                print("Error: Wrong gs_res_list length.")
                return
        elif len(gs_res_list) == len(gs_path_list):
            gaussians = GaussianModel(dataset.sh_degree)
            _gs_res = []
            for gs_path, gs_res_level in zip(gs_path_list, gs_res_list):
                obj = GaussianModel(dataset.sh_degree)
                obj.load_ply(gs_path)
                gaussians.merge(obj)
                _gs_res.extend([float(gs_res_level) for i in range(len(obj.get_xyz()))])
            gs_res = torch.tensor(_gs_res, device="cuda")
        else:
            print("Error: gs_res_list length must match gs_path_list length.")
            return
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), 
                       gaussians, gs_res,  
                       pipeline, background, dataset.train_test_exp, separate_sh)
            
        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), 
                       gaussians, gs_res, 
                       pipeline, background, dataset.train_test_exp, separate_sh)
            
            
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
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    print(float(args.gs_res_list[0]))
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE,
                # >>>> [YC] add
                gs_path_list=args.gs_path_list, gs_res_list=args.gs_res_list
                # <<<< [YC] add
                )