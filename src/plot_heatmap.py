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


data_dir1 = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/results/576_half-dla_34_mot17_half_hm/MOT17-02-SDP_hm/300.txt'
data_dir2 = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/results/640_half-dla_34_mot17_half_hm/MOT17-02-SDP_hm/300.txt'
data_dir3 = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/results/704_half-dla_34_mot17_half_hm/MOT17-02-SDP_hm/300.txt'
data_dir4 = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/results/864_half-dla_34_mot17_half_hm/MOT17-02-SDP_hm/300.txt'
data_dir5 = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/results/1088_half-dla_34_mot17_half_hm/MOT17-02-SDP_hm/300.txt'

data1 = np.loadtxt(data_dir1)
data2 = np.loadtxt(data_dir2)
data3 = np.loadtxt(data_dir3)
data4 = np.loadtxt(data_dir4)
data5 = np.loadtxt(data_dir5)

data1_new = _nms(torch.from_numpy(data1).unsqueeze(0).unsqueeze(0))
data2_new = _nms(torch.from_numpy(data2).unsqueeze(0).unsqueeze(0))
data3_new = _nms(torch.from_numpy(data3).unsqueeze(0).unsqueeze(0))
data4_new = _nms(torch.from_numpy(data4).unsqueeze(0).unsqueeze(0))
data5_new = _nms(torch.from_numpy(data5).unsqueeze(0).unsqueeze(0))

plt.subplot(5, 2, 1)
plt.imshow(data1, cmap='hot')
plt.subplot(5, 2, 2)
plt.imshow(data1_new.cpu().numpy().squeeze(), cmap='hot')

plt.subplot(5, 2, 3)
plt.imshow(data2, cmap='hot')
plt.subplot(5, 2, 4)
plt.imshow(data2_new.cpu().numpy().squeeze(), cmap='hot')

plt.subplot(5, 2, 5)
plt.imshow(data3, cmap='hot')
plt.subplot(5, 2, 6)
plt.imshow(data3_new.cpu().numpy().squeeze(), cmap='hot')

plt.subplot(5, 2, 7)
plt.imshow(data4, cmap='hot')
plt.subplot(5, 2, 8)
plt.imshow(data4_new.cpu().numpy().squeeze(), cmap='hot')

plt.subplot(5, 2, 9)
plt.imshow(data4, cmap='hot')
plt.subplot(5, 2, 10)
plt.imshow(data4_new.cpu().numpy().squeeze(), cmap='hot')

plt.show()
plt.savefig('heatmaps_new.jpg')