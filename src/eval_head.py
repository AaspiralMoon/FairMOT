# this script is for evaluating the multiknob head
# Author: Renjie Xu
# Time: 2023/2/24

import _init_paths
import torch
import os
import os.path as osp
import numpy as np

from models.model import create_model, load_model
import datasets.dataset.jde as datasets
from models.decode import mot_decode
from utils.post_process import ctdet_post_process
from models.decode import _topk,_nms
from utils.image import transform_preds

# heatmap = torch.rand((1, 3, 4, 4))
# scores, inds, clses, ys, xs = _topk(heatmap, K=5)
# print('heatmap is: ', heatmap)
# print('scores is: ', scores)
# print('inds is: ', inds)
# print('clses is: ', clses)
# print('ys is: ', ys)
# print('xs is: ', xs)

def hadamard_operation(A, B): # Element-wise Hadamard product
    return A * B 

def compare_hms(hm, hm_knob):
    hm = hm.squeeze()                       
    hm_knob = hm_knob.squeeze(0)
    for i in range(hm_knob.shape[0]):
        print(torch.div(torch.sum(hadamard_operation(hm, hm_knob[i])), torch.sum(hm)))
    return 

model_path = '/nfs/u40/xur86/projects/DeepScale/FairMOT/exp/mot_multiknob/multiknob_with_pretrain/model_last.pth'
data_path = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17_multiknob/train/MOT17-02-SDP/img1'
output_root = '/nfs/u40/xur86/projects/DeepScale/FairMOT/exp/mot_multiknob/'

arch = 'full-dla_34'
heads = {'hm': 1, 'hmknob': 75, 'wh': 4, 'id': 128, 'reg': 2}
head_conv = 256

model = create_model(arch, heads, head_conv)
model = load_model(model, model_path)
model = model.to(torch.device('cuda'))
model.eval()

dataloader = datasets.LoadImages(data_path, (1088, 608))
len_all = len(dataloader)
start_frame = int(len_all / 2)

for i, (path, img, img0) in enumerate(dataloader):
    if i < start_frame:
        continue
    blob = torch.from_numpy(img).cuda().unsqueeze(0)
    with torch.no_grad():
        output = model(blob)[-1]
        hm = output['hm'].sigmoid_()                              # heatmaps
        hm_knob = output['hmknob'].sigmoid_()                     # heatmaps for different knobs
        hm = _nms(hm)
        hm_knob = _nms(hm_knob)
        # compare_hms(hm, hm_knob)
        hm = hm.squeeze()                       
        hm_knob = hm_knob.squeeze(0)
        # mask = hm > 0.1
        # points = torch.where(mask)
        # mask2 = hm_knob[1] > 0.1
        # points2 = torch.where(mask2)
        # print(points)
        # print(points2)
    break
