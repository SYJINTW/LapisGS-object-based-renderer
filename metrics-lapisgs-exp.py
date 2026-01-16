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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import l1_loss, ssim, l1_loss_mask
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr, masked_psnr
from argparse import ArgumentParser

from pytorch_msssim_mask import ssim as ssim_mask
 
def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def readMask(mask_path):
    # 1. Open as Grayscale ("L")
    # This reads the brightness: Black=0, White=255
    img = Image.open(mask_path).convert("L")
    
    # 2. Convert to Tensor
    # tf.to_tensor automatically scales [0, 255] -> [0.0, 1.0]
    # So White becomes 1.0, Black becomes 0.0
    tensor = tf.to_tensor(img).cuda()
    
    # 3. Strict Binarization
    # Any value > 0.5 becomes exactly True
    # Any value <= 0.5 becomes exactly False
    binary_mask = tensor > 0.5
    
    # 4. Convert to Float (0.0 and 1.0) or Int (0 and 1)
    # .float() makes it 0.0 and 1.0 (Best for multiplication)
    # .long() makes it 0 and 1 (Best for indexing)
    mask = binary_mask.float() 
    
    # Optional: Unsqueeze to add batch dimension if needed [1, 1, H, W]
    mask = mask.unsqueeze(0)
    
    return mask

def evaluate(model_paths, cam_name, 
            gt_dir, mask_dir, output_dir):
    gt_dir = Path(gt_dir)
    print("GT dir:", gt_dir)
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            # test_dir = Path(scene_dir) / "test"
            test_dir = Path(scene_dir) / cam_name  # [YC] change
            
            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                # gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                masks = []
                for image_name in image_names:
                    # print(Path(mask_dir) / f"r_{int(image_name.split('.')[0])}_mask.png")
                    masks.append(readMask(Path(mask_dir) / f"r_{int(image_name.split('.')[0])}_mask.png"))

                ssims = []
                masked_ssims = []
                psnrs = []
                masked_psnrs = []
                lpipss = []
                losses = []
                masked_losses = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    masked_ssims.append(ssim_mask(renders[idx], gts[idx], 
                                                  mask=masks[idx].repeat(1, 3, 1, 1).bool(),
                                                  data_range=1.0,
                                                  size_average=True).item())
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    masked_psnrs.append(masked_psnr(renders[idx], gts[idx], masks[idx]))
                    # lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
                    lpipss.append(-1.0)
                    # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
                    losses.append(0.8 * l1_loss(renders[idx], gts[idx]) + 0.2 * (1.0 - ssim(renders[idx], gts[idx])))
                    masked_losses.append(0.8 * l1_loss_mask(renders[idx], gts[idx], masks[idx]) + 0.2 * (1.0 - ssim(renders[idx], gts[idx])))
                    
                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  MASKED_SSIM : {:>12.7f}".format(torch.tensor(masked_ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  MASKED_PSNR : {:>12.7f}".format(torch.tensor(masked_psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("  LOSS: {:>12.7f}".format(torch.tensor(losses).mean(), ".5"))
                print("  MASKED_LOSS: {:>12.7f}".format(torch.tensor(masked_losses).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                    "MASKED_SSIM": torch.tensor(masked_ssims).mean().item(),
                                                    "PSNR": torch.tensor(psnrs).mean().item(),
                                                    "MASKED_PSNR": torch.tensor(masked_psnrs).mean().item(),
                                                    "LPIPS": torch.tensor(lpipss).mean().item(),
                                                    "LOSS": torch.tensor(losses).mean().item(),
                                                    "MASKED_LOSS": torch.tensor(masked_losses).mean().item(),
                                                    "SSIM_std": torch.tensor(ssims).std().item(),
                                                    "MASKED_SSIM_std": torch.tensor(masked_ssims).std().item(),
                                                    "PSNR_std": torch.tensor(psnrs).std().item(),
                                                    "MASKED_PSNR_std": torch.tensor(masked_psnrs).std().item(),
                                                    "LPIPS_std": torch.tensor(lpipss).std().item(),
                                                    "LOSS_std": torch.tensor(losses).std().item(),
                                                    "MASKED_LOSS_std": torch.tensor(masked_losses).std().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                        "MASKED_SSIM": {name: mssim for mssim, name in zip(torch.tensor(masked_ssims).tolist(), image_names)},
                                                        "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                        "MASKED_PSNR": {name: mpsnr for mpsnr, name in zip(torch.tensor(masked_psnrs).tolist(), image_names)},
                                                        "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                        "LOSS": {name: loss for loss, name in zip(torch.tensor(losses).tolist(), image_names)},
                                                        "MASKED_LOSS": {name: mloss for mloss, name in zip(torch.tensor(masked_losses).tolist(), image_names)}})
            if output_dir != "":
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                with open(Path(output_dir) / f"results_{cam_name}.json", 'w') as fp:
                    json.dump(full_dict[scene_dir], fp, indent=True)
                with open(Path(output_dir) / f"per_view_{cam_name}.json", 'w') as fp:
                    json.dump(per_view_dict[scene_dir], fp, indent=True)
            else:
                with open(scene_dir + f"/results_{cam_name}.json", 'w') as fp:
                    json.dump(full_dict[scene_dir], fp, indent=True)
                with open(scene_dir + f"/per_view_{cam_name}.json", 'w') as fp:
                    json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument("--cam_name", type=str, default="test", help="Name of our cam for rendering results name.")
    parser.add_argument('--gt_dir', type=str, default="")
    parser.add_argument('--mask_dir', type=str, default="")
    parser.add_argument('--output_dir', type=str, default="")
    args = parser.parse_args()
    evaluate(args.model_paths, args.cam_name, 
            args.gt_dir, args.mask_dir,
            args.output_dir)
