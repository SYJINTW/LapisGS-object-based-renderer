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

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def masked_psnr(img1, img2, mask):
    # print(img1.shape, img2.shape, mask.shape)
    mask_expand = mask.expand_as(img1)
    diff = (img1 - img2) * mask_expand
    numerator = (diff ** 2).reshape(img1.shape[0], -1).sum(1, keepdim=True)
    denominator = mask_expand.reshape(mask.shape[0], -1).sum(1, keepdim=True)
    
    # 計算 MSE (加上 epsilon 防止除以 0)
    mse = numerator / (denominator + 1e-8)
    
    # 4. 計算 PSNR
    return 20 * torch.log10(1.0 / (torch.sqrt(mse) + 1e-8))