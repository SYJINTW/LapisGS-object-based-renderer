from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

from PIL import Image
import torchvision.transforms.functional as tf

def readImage(render_path, gt_path):
    render = Image.open(render_path)
    gt = Image.open(gt_path)
    render = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
    gt = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()
    return render, gt

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

render, gt = readImage("/home/syjintw/Desktop/NUS/lapisgs-output/lego/opacity/lego_res8/ours_1x_factor8/ours_30000/renders/00031.png", 
                       "/home/syjintw/Desktop/NUS/lapisgs-output/lego/opacity/lego_res8/ours_1x_factor8/ours_30000/gt/00031.png")
ssim_val = ssim( render, gt, data_range=1.0, size_average=False) # return (N,)
print("SSIM:", ssim_val.item())

mask = readMask("/home/syjintw/Desktop/NUS/tmp/lego/results_test_1x/mask/r_31_mask.png")
mask = mask.repeat(1, 3, 1, 1).bool()
ssim_val = ssim( render, gt, data_range=1.0, size_average=True,
                mask=mask) # return (N,)
print("SSIM:", ssim_val.item())