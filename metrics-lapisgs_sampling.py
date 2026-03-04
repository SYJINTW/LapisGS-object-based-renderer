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
import numpy as np

from pytorch_msssim_mask import ssim as ssim_mask
 
# def readImages(renders_dir, gt_dir):
#     renders = []
#     gts = []
#     image_names = []
#     for fname in os.listdir(renders_dir):
#         render = Image.open(renders_dir / fname)
#         gt = Image.open(gt_dir / fname)
#         renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
#         gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
#         image_names.append(fname)
#     return renders, gts, image_names

def readImages(image_dir, fname_list):
    images = []
    for fname in fname_list:
        render = Image.open(image_dir / fname)
        images.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
    return images

def get_binary_mask(black_render, white_render, threshold=0.01):
    """
    透過比較黑白背景的渲染圖來產生 Alpha Mask。
    
    輸入: 
        black_render: Tensor [3, H, W] 或 [1, 3, H, W]
        white_render: Tensor [3, H, W] 或 [1, 3, H, W]
    輸出: 
        mask: Tensor [1, 1, H, W], 數值只有 0.0 或 1.0 (Binary)
    """
    # --- 1. 維度處理 (防呆) ---
    # 如果輸入來自 readImages，通常會是 [1, 3, H, W]，我們需要先 squeeze 掉 batch 維度
    # 變成 [3, H, W] 方便做 dim=0 的 max 計算
    if black_render.dim() == 4:
        black_render = black_render.squeeze(0)
    if white_render.dim() == 4:
        white_render = white_render.squeeze(0)

    # --- 2. 計算絕對差異 ---
    diff = torch.abs(white_render - black_render) # [3, H, W]
    
    # --- 3. 跨通道 (RGB) 取最大值 ---
    # 邏輯：只要 R, G, B 任一通道有變動，該 Pixel 就視為前景
    diff_max, _ = torch.max(diff, dim=0) # [3, H, W] -> [H, W]
    
    # --- 4. 二值化 (Binary) ---
    # 大於 threshold 變 1.0，小於變 0.0
    mask = (diff_max > threshold).float() # [H, W]
    
    # --- 5. 重塑維度為 [1, 1, H, W] ---
    # [H, W] -> [1, H, W] (Channel) -> [1, 1, H, W] (Batch)
    mask = mask.unsqueeze(0).unsqueeze(0)
    
    return mask

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

def cal_mask_from_blackwhite_numpy(black_render, white_render):
    """
    使用 NumPy 在 CPU 上進行 Pixel-wise 遮罩計算。
    
    Args:
        black_image: Tensor (C, H, W) 或 NumPy (H, W, C)
        white_image: Tensor (C, H, W) 或 NumPy (H, W, C)
        threshold: 容忍值。如果 (White - Black) <= threshold，則視為背景。
                   建議設一點值 (例如 10/255) 來過濾渲染雜訊。
    """
    
    diff = torch.abs(white_render - black_render)
    mask_1ch = (diff > 0.02).any(dim=0, keepdim=True).float()
    
    return mask_1ch

def evaluate(black_gt_dir, 
            white_gt_dir, dist_dir, 
            output_dir):
    
    black_gt_dir = Path(black_gt_dir)
    white_gt_dir = Path(white_gt_dir)
    dist_dir = Path(dist_dir)
    full_dict = {}
    per_view_dict = {}
    
    fname_list = [fname for fname in os.listdir(black_gt_dir)]
    black_gts = readImages(black_gt_dir, fname_list)
    white_gts = readImages(white_gt_dir, fname_list)    
    renders = readImages(dist_dir, fname_list)
    
    masks = []
    for black_render, white_render in zip(black_gts, white_gts):
        mask = get_binary_mask(black_render, white_render)
        masks.append(mask)
    
    valid_pixels = []
    ssims = []
    masked_ssims = []
    psnrs = []
    masked_psnrs = []
    lpipss = []
    losses = []
    masked_losses = []

    for idx in tqdm(range(len(black_gts)), desc="Metric evaluation progress"):
        valid_pixels.append(torch.sum(masks[idx]).item())
        ssims.append(ssim(black_gts[idx], renders[idx]))
        masked_ssims.append(ssim_mask(black_gts[idx], renders[idx], 
                                        mask=masks[idx].repeat(1, 3, 1, 1).bool(),
                                        data_range=1.0,
                                        size_average=True).item())
        psnrs.append(psnr(black_gts[idx], renders[idx]))
        masked_psnrs.append(masked_psnr(black_gts[idx], renders[idx], masks[idx]))
        # lpipss.append(lpips(black_gts[idx], renders[idx], net_type='vgg'))
        lpipss.append(-1.0)
        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        losses.append(0.8 * l1_loss(black_gts[idx], renders[idx]) + 0.2 * (1.0 - ssim(black_gts[idx], renders[idx])))
        masked_losses.append(0.8 * l1_loss_mask(black_gts[idx], renders[idx], masks[idx]) + 0.2 * (1.0 - ssim(black_gts[idx], renders[idx])))
        
    print("  VALID_PIXELS: {:>12.7f}".format(torch.tensor(valid_pixels).mean(), ".5"))
    print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
    print("  MASKED_SSIM : {:>12.7f}".format(torch.tensor(masked_ssims).mean(), ".5"))
    print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
    print("  MASKED_PSNR : {:>12.7f}".format(torch.tensor(masked_psnrs).mean(), ".5"))
    print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
    print("  LOSS: {:>12.7f}".format(torch.tensor(losses).mean(), ".5"))
    print("  MASKED_LOSS: {:>12.7f}".format(torch.tensor(masked_losses).mean(), ".5"))
    print("")

    full_dict.update({
                    # mean
                    "VALID_PIXELS": torch.tensor(valid_pixels).mean().item(),
                    "SSIM": torch.tensor(ssims).mean().item(),
                    "MASKED_SSIM": torch.tensor(masked_ssims).mean().item(),
                    "PSNR": torch.tensor(psnrs).mean().item(),
                    "MASKED_PSNR": torch.tensor(masked_psnrs).mean().item(),
                    "LPIPS": torch.tensor(lpipss).mean().item(),
                    "LOSS": torch.tensor(losses).mean().item(),
                    "MASKED_LOSS": torch.tensor(masked_losses).mean().item(),
                    # std
                    "VALID_PIXELS_std": torch.tensor(valid_pixels).std().item(),
                    "SSIM_std": torch.tensor(ssims).std().item(),
                    "MASKED_SSIM_std": torch.tensor(masked_ssims).std().item(),
                    "PSNR_std": torch.tensor(psnrs).std().item(),
                    "MASKED_PSNR_std": torch.tensor(masked_psnrs).std().item(),
                    "LPIPS_std": torch.tensor(lpipss).std().item(),
                    "LOSS_std": torch.tensor(losses).std().item(),
                    "MASKED_LOSS_std": torch.tensor(masked_losses).std().item()
                    })
    
    per_view_dict.update({
                        "VALID_PIXELS": {name: vp for vp, name in zip(torch.tensor(valid_pixels).tolist(), fname_list)},
                        "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), fname_list)},
                        "MASKED_SSIM": {name: mssim for mssim, name in zip(torch.tensor(masked_ssims).tolist(), fname_list)},
                        "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), fname_list)},
                        "MASKED_PSNR": {name: mpsnr for mpsnr, name in zip(torch.tensor(masked_psnrs).tolist(), fname_list)},
                        "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), fname_list)},
                        "LOSS": {name: loss for loss, name in zip(torch.tensor(losses).tolist(), fname_list)},
                        "MASKED_LOSS": {name: mloss for mloss, name in zip(torch.tensor(masked_losses).tolist(), fname_list)}
                        })
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", 'w') as fp:
        json.dump(full_dict, fp, indent=True)
    with open(output_dir / "per_view.json", 'w') as fp:
        json.dump(per_view_dict, fp, indent=True)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    # parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument("--cam_name", type=str, default="test", help="Name of our cam for rendering results name.")
    parser.add_argument('--black_gt_dir', type=str, default="")
    parser.add_argument('--white_gt_dir', type=str, default="")
    parser.add_argument('--dist_dir', type=str, default="")
    # parser.add_argument('--mask_dir', type=str, default="")
    parser.add_argument('--output_dir', type=str, default="")
    args = parser.parse_args()
    evaluate(args.black_gt_dir, args.white_gt_dir, 
            args.dist_dir,
            args.output_dir)
