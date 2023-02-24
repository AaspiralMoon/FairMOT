# this script is for generating and saving detection results
# Author: Renjie Xu
# Time: 2023/2/22

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch

from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets

from tracking_utils.utils import mkdir_if_missing
from opts import opts

# python track_half.py --load_model /nfs/u40/xur86/projects/DeepScale/FairMOT/exp/mot/new_mot17_dla34/model_last.pth --arch dla_34 --gen_hm --gen_dets

def eval_seq(opt, dataloader, output_root):
    tracker = JDETracker(opt)
    len_all = len(dataloader)
    start_frame = int(len_all / 2)
    for i, (path, img, img0) in enumerate(dataloader):
        if i >= start_frame:
            continue

        # run tracking
        blob = torch.from_numpy(img).cuda().unsqueeze(0)

        if opt.gen_dets:
            dets, online_targets = tracker.update(blob, img0)
            with open(osp.join(output_root, '{}.txt'.format(int(i+1))), 'w+') as f:
                np.savetxt(f, dets, '%.4f')


def main(opt, data_root, output_root):
    mkdir_if_missing(output_root)

    # run tracking
    dataloader = datasets.LoadImages(data_root, opt.img_size)
    eval_seq(opt, dataloader, output_root)


if __name__ == '__main__':
    torch.cuda.set_device(2)
    opt = opts().init()
    data_root = opt.data_dir
    output_root = opt.output_root
    main(opt,
         data_root=data_root,
         output_root=output_root)
