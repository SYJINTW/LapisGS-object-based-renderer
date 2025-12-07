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

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    far_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "far_renders")
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(far_render_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        # Drop resolution by half
        view.image_height = view.image_height // 2
        view.image_width = view.image_width // 2
        
        # bg_color_template = [0, 0, 0] # black background
        bg_color_template = [1, 1, 1] # white background
        bg_color = torch.tensor(bg_color_template, dtype=torch.float32, device="cuda").view(3, 1, 1)
        bg_color = bg_color.expand(3, view.image_height, view.image_width)
        bg_depth = torch.full((1, view.image_height, view.image_width), 0, dtype=torch.float32, device="cuda")
    
        gt = view.original_image[0:3, :, :]
        
        # Far + bg
        far_rendering = render(view, gaussians, pipeline, 
                               bg_color, bg_depth,
                               far_thres=100.0, near_thres=4.0,
                               use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]


        print("type of far_rendering:", type(far_rendering))
        print("shape of far_rendering:", far_rendering.shape)
        
        # Add batch dimension: [3, 400, 400] -> [1, 3, 400, 400]
        input_tensor = far_rendering.unsqueeze(0)
        
        # Interpolate (Scale)
        # mode='bilinear' is standard for images (smooths values). 
        # mode='nearest' preserves exact pixel values (blocky).
        scaled_tensor = F.interpolate(input_tensor, size=(800, 800), mode='bilinear', align_corners=False)

        # Remove batch dimension: [1, 3, 800, 800] -> [3, 800, 800]
        far_rendering_scaled = scaled_tensor.squeeze(0)

        # Double resolution
        view.image_height = view.image_height * 2
        view.image_width = view.image_width * 2
        
        # Near + Far + bg
        rendering = render(view, gaussians, pipeline, 
                           far_rendering_scaled, bg_depth,
                           far_thres=4.0, near_thres=-1.0,
                           use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
        
        # rendering = render(view, gaussians, pipeline, 
        #                    bg_color, bg_depth,
        #                    far_thres=100.0, near_thres=-1.0,
        #                    use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
        
        if args.train_test_exp:
            far_rendering = far_rendering[..., far_rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]
        
        torchvision.utils.save_image(far_rendering, os.path.join(far_render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        
        if idx >= 5:
            break  # [YC] debug: only render one image
    
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)