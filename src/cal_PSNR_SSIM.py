# This script is for calculating the PSNR and SSIM of an image
# Author: Renjie Xu
# Time: 2023/3/23

# ffmpeg -i images/%06d.jpg -c:v libx264 -crf 0 output.mp4
# ffmpeg -i output.mp4 -c:v libx264 -crf 10 output_10.mp4
# ffmpeg -i output_10.mp4 output_10/%06d.jpg

import numpy as np
import imageio
from skimage import img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize

# Replace 'image1_path' and 'image2_path' with the paths to your own image files
image1_path = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17_multiknob/MOT17-02-SDP/QP_0/images/000001.jpg'
image2_path = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17_multiknob/MOT17-02-SDP/QP_50/images/000001.jpg'

# Load the images
img1 = img_as_float(imageio.imread(image1_path))
img2 = img_as_float(imageio.imread(image2_path))

# Resize the images to the same shape if they have different shapes
if img1.shape != img2.shape:
    img1 = resize(img1, img2.shape)

# Ensure the images have the same shape and dtype
assert img1.shape == img2.shape and img1.dtype == img2.dtype, "Images must have the same shape and dtype"

# Compute PSNR
psnr_value = psnr(img1, img2, data_range=img1.max() - img1.min())

# Compute SSIM
ssim_value = ssim(img1, img2, multichannel=True)

print(f"PSNR: {psnr_value}")
print(f"SSIM: {ssim_value}")
