# This script is for calculating the detection rate of different QPs
# Author: Renjie Xu
# Time: 2023/4/16
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from models.decode import _nms

def heatmap_to_binary(heatmap, threshold):
    binary = (heatmap > threshold).to(torch.float32)
    return binary

def hadamard_operation(A, B): # Element-wise Hadamard product
    return A * B 

def compare_hms(hm, hm_gt):
    if hm.size() != hm_gt.size():
        # Rescale hm to match the size of hm_gt
        hm = F.interpolate(hm, size=hm_gt.shape[-2:], mode='bilinear', align_corners=False)
    hm = _nms(hm)
    hm_gt = _nms(hm_gt)
    hm = heatmap_to_binary(hm, 0.4)       
    hm_gt = heatmap_to_binary(hm_gt, 0.4)
    hm = hm.squeeze()
    hm_gt = hm_gt.squeeze()
    det_rate = torch.div(torch.sum(hadamard_operation(hm, hm_gt)), torch.sum(hm_gt))
    return det_rate.item()

# def count_det(file_path):                 # count the detection numbers, i.e., the number of lines in the txt
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#     return len(lines)

result_root = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/results'
imgsz_list = [1088, 864, 704, 640, 576]
model_list = ['full']
exp_id_list = []
for imgsz in imgsz_list:
    for m in model_list:
        exp_id_list.append('{}_{}_dla_hm'.format(imgsz, m))

seqs = ['MOT17-02-SDP',
        'MOT17-04-SDP',
        'MOT17-05-SDP',
        'MOT17-09-SDP',
        'MOT17-10-SDP',
        'MOT17-11-SDP',
        'MOT17-13-SDP']

hm_dict = {}
for exp_id in exp_id_list:
    exp_hm_dict = {}
    exp_path = osp.join(result_root, exp_id)
    for seq in seqs:
        seq_hm_list = []
        hm_path = osp.join(exp_path, '{}_hm'.format(seq))
        for file in os.listdir(hm_path):
            hm = torch.load(osp.join(hm_path, file))
            seq_hm_list.append(hm)
        exp_hm_dict['{}'.format(seq)] = seq_hm_list
    hm_dict['{}'.format(exp_id)] = exp_hm_dict


det_rate_dict = {}
hm_gt = hm_dict['{}'.format(exp_id_list[0])]         # use the most expensive configuration as gt
for exp_id in exp_id_list[1:]:
    exp_det_rate_dict = {}
    hm = hm_dict['{}'.format(exp_id)]
    for seq in seqs:
        hm_gt_seq = hm_gt['{}'.format(seq)]
        hm_seq = hm['{}'.format(seq)]
        seq_det_rate_list = [compare_hms(x, y) for x, y in zip(hm_seq, hm_gt_seq)]
        exp_det_rate_dict['{}'.format(seq)] = seq_det_rate_list
    det_rate_dict['{}'.format(exp_id)] = exp_det_rate_dict

# Create a 3x2 grid of subplots
fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(30, 40))
axes = axes.flatten()  # Flatten the axes array for easy indexing

# Define colors for each line
colors = ['red', 'blue', 'green', 'orange']

# Plot the line plots for each dataset
for i, ax in enumerate(axes):
    if i < len(seqs):
        for j, exp_id in enumerate(exp_id_list[1:]):
            ax.plot(list(range(len(det_rate_dict[exp_id][seqs[i]]) + 1, 2 * len(det_rate_dict[exp_id][seqs[i]]) + 1)), det_rate_dict[exp_id][seqs[i]], label='{}'.format(exp_id), color=colors[j])
        ax.set_title('{}'.format(seqs[i]))
        ax.set_xlabel('Frames')
        ax.set_ylabel('Detection Rate')
        ax.legend()


# Adjust the layout
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# Save the plot as an image file
plt.savefig('detection rate.png', dpi=300)