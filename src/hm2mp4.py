import os
import os.path as osp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def mkdir_if_missing(d):
    if not osp.exists(d):
        os.makedirs(d)


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep



path = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/results/864_hm/MOT17-11-SDP_hm'
save_path = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/results/864_hm/MOT17-11-SDP_hm_img'

mkdir_if_missing(save_path)
file_list = os.listdir(path)

for file in file_list:
    file = file.strip().split('.')[0]
    img = np.loadtxt(os.path.join(path, '{}.txt'.format(file)))
    img = _nms(torch.from_numpy(img).unsqueeze(0).unsqueeze(0))
    img = img.cpu().numpy().squeeze()
    plt.imshow(img, cmap='hot')
    plt.savefig(os.path.join(save_path, '{}.jpg'.format(file)))

cmd_str = 'cd {}; ffmpeg -f image2 -r 30 -start_number {} -i %03d.jpg video.avi'.format(save_path, file_list[1].strip().split('.')[0])
os.system(cmd_str)
print(cmd_str)