# this script is for testing any code
# Author: Renjie Xu
# Time: 2023/2/22

import _init_paths
import torch
import os
import os.path as osp
from models.model import create_model, load_model
import datasets.dataset.jde as datasets

model_path = '/nfs/u40/xur86/projects/DeepScale/FairMOT/models/ctdet_coco_dla_2x.pth'
data_path = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17_multiknob/train/MOT17-02-SDP/img1'

arch = 'full-dla_34'
heads = {'hm': 1, 'wh': 4, 'id': 128, 'reg': 2}
head_conv = 256


model = create_model(arch, heads, head_conv)
model = load_model(model, model_path)
model = model.to(torch.device('cuda'))
model.eval()

# dataloader = datasets.LoadImages(data_path, (1088, 608))

# for i, (path, img, img0) in enumerate(dataloader):
#     blob = torch.from_numpy(img).cuda().unsqueeze(0)
#     output = model(blob)[-1]
#     hm_knob = output['hm']

#     print(hm_knob.shape)