'''
    process evaluation results of mot datasets: frame_id, obj_id, x, y, w, h

'''
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
import math

from opts import opts

def str2np(dir, seq):
    data_np = []
    with open(os.path.join(dir, '{}'.format(seq), 'train', '{}_labeled.txt'.format(seq)), 'r') as f:
        lines = f.readlines()
        for line in lines:
            data_np.append(line.strip().split(','))
    return np.array(data_np, dtype=float)

def main(data_dir, base_dir, seqs):
    for seq in seqs:
        data = str2np(data_dir, seq)[1: -1, [0, 2, 3, 4]]        # remove frame_id, first row
        data_base = str2np(base_dir, seq)[1: -1, [0, 2, 3, 4]]   # remove frame_id, first row
        data[:, 0] = np.divide(data[:, 0], data_base[:, 0] + 1e-6)
        data = data[np.all(data>0, axis=1)]                     # remove rows containing 0
        with open(os.path.join(data_dir, 'train.txt'), 'a+') as f:
            np.savetxt(f, data, '%.4f')

if __name__ == '__main__':
    opt = opts().init()
    if opt.val_mot17:
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP'''
    seqs = [seq.strip() for seq in seqs_str.split()]
    base_dir = '/u40/xur86/datasets/MOT17/images/results/dla_34_mot17_half'
    data_dir = '/u40/xur86/datasets/MOT17/images/results/half-dla_34_mot17_half'
    main(data_dir=data_dir,
         base_dir=base_dir,
         seqs=seqs)