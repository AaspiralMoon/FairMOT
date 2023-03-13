# this script is for testing any code
# Author: Renjie Xu
# Time: 2023/2/22

import _init_paths
import torch
import os
import os.path as osp
import numpy as np
import matplotlib
import time
import matplotlib.pyplot as plt
import cv2

detection_result_path = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17_multiknob/results'
save_path = '/nfs/u40/xur86/projects/DeepScale/FairMOT/exp/mot_multiknob/verify_labels/verify_detections'

def plot_label(img, labels):
    img = cv2.imread(img)
    matplotlib.use('Agg')
    plt.close('all')
    plt.figure()
    plt.imshow(img[:, :, ::-1])
    plt.plot(labels[:, [0, 2, 2, 0, 0]].T, labels[:, [1, 1, 3, 3, 1]].T, '.-')
    plt.axis('off')
    plt.savefig(osp.join(save_path, 'test.jpg'))
    time.sleep(3)
    plt.close('all')

img = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17_multiknob/train/MOT17-13-SDP/img1/000001.jpg'
labels = np.loadtxt('/nfs/u40/xur86/projects/DeepScale/datasets/MOT17_multiknob/results/MOT17-13-SDP/576_quarter/1.txt')
plot_label(img, labels)