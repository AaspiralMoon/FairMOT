# This script is for calculating the PSNR and SSIM of an image
# Author: Renjie Xu
# Time: 2023/3/23

import numpy as np
from skimage import data, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Load or read your two RGB images (img1 and img2) as NumPy arrays
# For the sake of example, we'll use two built-in images from skimage
img1 = img_as_float(data.astronaut())
img2 = img_as_float(data.camera())

# Ensure the images have the same shape and dtype
assert img1.shape == img2.shape and img1.dtype == img2.dtype, "Images must have the same shape and dtype"

# Compute PSNR
psnr_value = psnr(img1, img2, data_range=img1.max() - img1.min())

# Compute SSIM
ssim_value = ssim(img1, img2, multichannel=True)

print(f"PSNR: {psnr_value}")
print(f"SSIM: {ssim_value}")
