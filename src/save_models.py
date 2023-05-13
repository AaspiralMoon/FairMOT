from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import _init_paths

from opts import opts
from models.model import create_model

def save_model_with_model_key(path, epoch, model):
    data = {'epoch': epoch,
            'model': model}
    torch.save(data, path)

opt = opts().init()

# Step 1: Create the model
model = create_model(opt.arch, opt.heads, opt.head_conv) 

# Step 2: Load the saved checkpoint and extract the state_dict
ckpt_path = '/nfs/u40/xur86/projects/DeepScale/CenterNet/exp/ctdet/coco_half-yolo/model_best.pth'
ckpt = torch.load(ckpt_path)
state_dict = ckpt['state_dict']

# Step 3: Load the state_dict into the model
model.load_state_dict(state_dict)

# Step 4: Save the model again with the 'model' key
save_model_with_model_key('/nfs/u40/xur86/projects/DeepScale/FairMOT/models/yolov5s_half.pt', ckpt['epoch'], model)
