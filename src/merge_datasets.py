from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from cv2 import threshold

import _init_paths
import os
import os.path as osp
import cv2
# import logging
# import argparse
# import motmetrics as mm
import numpy as np
import torch
import torch.nn as nn
# import math
# import datasets.dataset.jde as datasets

from opts import opts

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def cal_hm(hm_r, hm_base, threshold):
    hm_r = _nms(torch.from_numpy(hm_r).unsqueeze(0).unsqueeze(0))
    hm_base = _nms(torch.from_numpy(hm_base).unsqueeze(0).unsqueeze(0))
    hm_r = hm_r.squeeze()
    hm_base = hm_base.squeeze()
    hm_r[hm_r > threshold] = 1
    hm_r[hm_r <= threshold] = 0
    hm_base[hm_base > threshold] = 1
    hm_base[hm_base <= threshold] = 0
    hm_base = torch.nn.functional.interpolate(hm_base.unsqueeze(0).unsqueeze(0), size=hm_r.shape)
    error = torch.div(torch.sum(torch.mul(hm_r, hm_base.squeeze())), torch.sum(hm_base.squeeze()) + 1e-6)
    return error.item()

def cal_dets(det_r, det_base):
    if det_base == 0:
        det_base = 1e-6
    return det_r/det_base

def main(result_root, result_filename, seqs, exps, threshold):
    results = []

    for i, exp in enumerate(exps[0: len(exps)-1]):         # i means discrete resolution
        hm = []
        det = []
        mAP_r = np.loadtxt(osp.join(result_root, exp, 'mAP.txt'))[0:2284].reshape(-1, 1)
        mAP_base = np.loadtxt(osp.join(result_root, exps[len(exps)-1], 'mAP.txt'))[0:2284].reshape(-1, 1)
        mAP = mAP_r/(mAP_base + 1e-6)
        mAP = mAP.reshape(-1, 1)
        for seq in seqs:
            hm_dir = osp.join(result_root, exp, '{}_hm'.format(seq))
            base_hm_dir = osp.join(result_root, exps[len(exps)-1], '{}_hm'.format(seq))
            det_dir = osp.join(result_root, exp, '{}_dets'.format(seq))
            base_det_dir = osp.join(result_root, exps[len(exps)-1], '{}_dets'.format(seq))
            for txtname in os.listdir(hm_dir):
                hm_r = np.loadtxt(osp.join(hm_dir, txtname))
                hm_base = np.loadtxt(osp.join(base_hm_dir, txtname))
                hm_error = cal_hm(hm_r, hm_base, threshold)
                hm = np.append(hm, hm_error)
            for txtname in os.listdir(det_dir):
                det_r = np.loadtxt(osp.join(det_dir, txtname))
                det_base = np.loadtxt(osp.join(base_det_dir, txtname))
                det_error = cal_dets(det_r, det_base)
                det = np.append(det, det_error)
        hm = hm.reshape(-1,1)
        det = det.reshape(-1,1)
        result = np.append(hm, det, axis=1)
        result = np.append(result, i*np.ones(hm.shape), axis=1)
        result = np.append(mAP, result, axis=1)
        results = np.append(results, result)
    results = results.reshape(-1, 4)
    with open(result_filename, 'w+') as f:
        np.savetxt(f, results, '%.4f')



if __name__ == '__main__':
    seqs_str = '''    MOT17-02-SDP
                    MOT17-04-SDP
                    MOT17-05-SDP
                    MOT17-09-SDP
                    MOT17-10-SDP
                    MOT17-11-SDP'''
    exps_str = '''576_half-dla_34_mot17_half_hm
                  640_half-dla_34_mot17_half_hm 
                  704_half-dla_34_mot17_half_hm
                  864_half-dla_34_mot17_half_hm 
                  1088_half-dla_34_mot17_half_hm'''            # change params here
    seqs = [seq.strip() for seq in seqs_str.split()]
    exps = [exp.strip() for exp in exps_str.split()]
    result_root = '/u40/xur86/projects/DeepScale/datasets/MOT17/images/results'
    result_filename = '/u40/xur86/projects/DeepScale/datasets/MOT17/images/results/half-dla_34_mot17_half_dets.txt'
    main(result_root=result_root,
         result_filename=result_filename,
         seqs=seqs,
         exps=exps,
         threshold=1e-2)