import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


data_dir = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/results/full_hm/MOT17-02-SDP_hm/305.txt'

data = np.loadtxt(data_dir)


data_new = _nms(torch.from_numpy(data).unsqueeze(0).unsqueeze(0))


plt.imshow(data, cmap='hot')
plt.savefig('02_305.jpg')