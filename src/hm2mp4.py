import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep



path = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/results/1088_dla_34_mot17_half_hm/MOT17-02-SDP_hm'
save_path = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/results/1088_dla_34_mot17_half_hm/MOT17-02-SDP-video'
file_list = os.listdir(path)

for file in file_list:
    file = file.strip().split('.')[0]
    img = np.loadtxt(os.path.join(path, '{}.txt'.format(file)))
    img = _nms(torch.from_numpy(img).unsqueeze(0).unsqueeze(0))
    img = img.cpu().numpy().squeeze()
    plt.imshow(img, cmap='hot')
    plt.savefig(os.path.join(save_path, '{}.jpg'.format(file)))