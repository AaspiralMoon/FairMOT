# this script is for evaluating the multiknob head
# Author: Renjie Xu
# Time: 2023/2/24

import _init_paths
import torch
import time
import os
import os.path as osp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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

def mkdir_if_missing(d):
    if not osp.exists(d):
        os.makedirs(d)
        
def plot_and_save_img(output_root, img, idx=-1):
    if idx == -1:
        mpimg.imsave(osp.join(output_root, 'heatmap.png'), img, cmap='hot')
    else:
        mpimg.imsave(osp.join(output_root, 'heatmap_{}.png'.format(idx)), img, cmap='hot')

def heatmap_to_binary(heatmap, threshold):
    binary = (heatmap > threshold).to(torch.float32)
    return binary

def plot_and_save_heatmaps(img_id, output_root, hm, hm_knob):
    hm = hm.squeeze().cpu().numpy()                   
    hm_knob = hm_knob.squeeze(0).cpu().numpy()
    output_path = osp.join(output_root, str(img_id))
    mkdir_if_missing(output_path)
    plot_and_save_img(output_path, hm)
    for i in range(len(hm_knob)):
        plot_and_save_img(output_path, hm_knob[i], i)

def hadamard_operation(A, B): # Element-wise Hadamard product
    return A * B 

def compare_hms(hm, hm_knob):
    hm = hm.squeeze()                       
    hm_knob = hm_knob.squeeze(0)
    for i in range(hm_knob.shape[0]):
        print('{} : '.format(knob_list[i]), torch.div(torch.sum(hadamard_operation(hm_knob[0], hm_knob[i])), torch.sum(hm_knob[0])))
    return 

model_path = '/nfs/u40/xur86/projects/DeepScale/FairMOT/exp/mot_multiknob/multiknob_res_and_model_lr_250_280/model_155.pth'
data_path = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17_multiknob/train/MOT17-04-SDP/img1'
output_root = '/nfs/u40/xur86/projects/DeepScale/FairMOT/exp/mot_multiknob/multiknob_res_and_model_2nd'
knob_list = []
imgsize_list = [1088, 864, 704, 640, 576]
model_list = ['full', 'half', 'quarter']
for imgsz in imgsize_list:
    for m in model_list:
        knob_list.append('{}_{}'.format(imgsz, m))

arch = 'full-dla_34'
heads = {'hm': 1, 'hmknob': 15, 'wh': 4, 'id': 128, 'reg': 2}
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
    print('Processing image: ', i + 1)
    blob = torch.from_numpy(img).cuda().unsqueeze(0)
    with torch.no_grad():
        output = model(blob)[-1]
        hm = output['hm'].sigmoid_()                              # heatmaps
        hm_knob = output['hmknob'].sigmoid_()                     # heatmaps for different knobs
        hm = _nms(hm)
        hm_knob = _nms(hm_knob)
        hm_binary = heatmap_to_binary(hm, 0.2)
        hm_knob_binary = heatmap_to_binary(hm_knob, 0.2)
        compare_hms(hm_binary, hm_knob_binary)
        plot_and_save_heatmaps(str(i+1), output_root, hm, hm_knob)
    break