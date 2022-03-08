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
import datasets.dataset.jde as datasets
from opts import opts

def str2np(dir, seq):
    data_np = []
    with open(os.path.join(dir, '{}'.format(seq), 'train', '{}_labeled.txt'.format(seq)), 'r') as f:
        lines = f.readlines()
        for line in lines:
            data_np.append(line.strip().split(','))
    return np.array(data_np, dtype=float)

def cal_hist(img, size=(112, 112), channel=2):  # b=0, g=1, r=2
    img = cv2.resize(img, size)
    hist = cv2.calcHist([img], [channel], None, [256], [0, 256])
    return hist

def cal_edge_sobel(img, size=(112, 112)):
    img = cv2.resize(img, size)
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    Scale_absX = cv2.convertScaleAbs(x)  
    Scale_absY = cv2.convertScaleAbs(y)
    Sobel = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
    return np.mean(Sobel)

def cal_edge_scharr(img, size=(112, 112)):
    img = cv2.resize(img, size)
    x = cv2.Scharr(img, cv2.CV_16S, 1, 0) 
    y = cv2.Scharr(img, cv2.CV_16S, 0, 1)
    Scale_absX = cv2.convertScaleAbs(x)
    Scale_absY = cv2.convertScaleAbs(y)
    Scharr = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
    return np.mean(Scharr)




def main(opt, data_root, result_root, seqs):
    for seq in seqs:
        dataloader = datasets.LoadImages(os.path.join(data_root, seq, 'img1'), opt.img_size)
        result_filename = os.path.join(result_root, '{}'.format(seq), 'hist_and_edge.txt')
        len_all = len(dataloader)
        start_frame = int(len_all / 2)
        result = []
        for i, (path, img, img0) in enumerate(dataloader):
            if i < start_frame:
                continue
            hist = cal_hist(img0)
            edge = cal_edge_sobel(img0)
            result.append((hist, edge))
        result = np.diff(result, axis=0)
        temp = []
        for hist, edge in result:
            hist = np.mean(abs(hist))
            edge = edge
            temp.append([hist, edge])
        temp = np.array(temp)
        data_labeled = str2np(result_root, seq)[:, [0, 2, 3, 4]]       # remove frame_id
        data_labeled = np.diff(data_labeled, axis=0)
        combined = np.concatenate([data_labeled, temp], axis=1)
        with open(result_filename, 'w+') as f:
            np.savetxt(f, combined, '%.4f')


        

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
    data_root = '/u40/xur86/datasets/MOT17/images/train'
    result_root = '/u40/xur86/datasets/MOT17/images/results/half-dla_34_mot17_half'
    main(opt,
         data_root=data_root,
         result_root=result_root,
         seqs=seqs)