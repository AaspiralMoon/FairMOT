# this script is for testing any code
# Author: Renjie Xu
# Time: 2023/2/22

import _init_paths
import torch
import os
import os.path as osp
import numpy as np
# import numpy as np
# import matplotlib
# import time
# import matplotlib.pyplot as plt
# import cv2

# detection_result_path = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17_multiknob/results'
# save_path = '/nfs/u40/xur86/projects/DeepScale/FairMOT/exp/mot_multiknob/verify_labels/verify_detections'

# def plot_label(img, labels):
#     img = cv2.imread(img)
#     matplotlib.use('Agg')
#     plt.close('all')
#     plt.figure()
#     plt.imshow(img[:, :, ::-1])
#     plt.plot(labels[:, [0, 2, 2, 0, 0]].T, labels[:, [1, 1, 3, 3, 1]].T, '.-')
#     plt.axis('off')
#     plt.savefig(osp.join(save_path, 'test.jpg'))
#     time.sleep(3)
#     plt.close('all')

# img = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17_multiknob/train/MOT17-13-SDP/img1/000001.jpg'
# labels = np.loadtxt('/nfs/u40/xur86/projects/DeepScale/datasets/MOT17_multiknob/results/MOT17-13-SDP/576_quarter/1.txt')
# plot_label(img, labels)

# avg_time = [0.5]
# avg_time_array = np.asarray(avg_time)
# result_root = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/results'
# path = osp.join(result_root, 'avg_fps.txt')
# print(avg_time_array)
# print(path)
# np.savetxt(path, 1.0 / avg_time_array, fmt='%.2f')

a = [11, 14, 8, 13, 10, 7, 5, 4, 12, 9, 2, 6, 1, 3, 0]
b = [2, 4, 5, 6, 10, 11, 12]

result = min((a.index(number), number) for number in b)[1]
aaa = [(a.index(number), number) for number in b]
print(min(aaa))
