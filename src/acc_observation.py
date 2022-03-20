from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import argparse
import torch
import json
import time
import os
import cv2

from scipy import interpolate
import numpy as np
from torchvision.transforms import transforms as T
from utils.utils import xywh2xyxy, ap_per_class, bbox_iou


def xywh2xyxy(x):
    # top-left x, y, w, h to x1, y1, x2, y2
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0])
    y[:, 1] = (x[:, 1])
    y[:, 2] = (x[:, 0] + x[:, 2])
    y[:, 3] = (x[:, 1] + x[:, 3])
    return y

def str2np(dir):
    data_np = []
    with open(dir, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data_np.append(line.strip().split(','))
    return np.array(data_np, dtype=float)

def load_and_bracketing(path, start, end):
    data = str2np(path)
    data = data[data[:, 0]>=start]
    data = data[data[:, 0]<=end]
    return data

def eval_F1(pred, gt, frame_rate):
    downratio = int(np.round(30/frame_rate))
    pred_list = []
    gt_list = []
    F1_scores = []
    frame_list = np.unique(gt[:, 0])[::downratio]
    for frame_id in frame_list:
        pred_list.append(pred[pred[:, 0]==frame_id])
        gt_list.append(gt[gt[:, 0]==frame_id])
    
    iou_thres = 0.5
   
    for pred, gt in zip(pred_list, gt_list):
        gt_boxes = torch.FloatTensor(xywh2xyxy(gt[:, 2:6]))
        pred_boxes = xywh2xyxy(pred[:, 2:6])
        correct = []
        detected = []
        for pred_bbox in pred_boxes:
            pred_bbox = torch.FloatTensor(pred_bbox).view(1, -1)
            # Compute iou with target boxes
            iou = bbox_iou(pred_bbox, gt_boxes, x1y1x2y2=True)[0]
            # Extract index of largest overlap
            best_i = np.argmax(iou)
            # If overlap exceeds threshold and classification is correct mark as correct
            if iou[best_i] > iou_thres and best_i not in detected:
                correct.append(1)
                detected.append(best_i)
            else:
                correct.append(0)

       #  Compute Average Precision (AP) per class
        AP, AP_class, R, P = ap_per_class(tp=correct,
                                            conf=pred[:, 6],
                                            pred_cls=np.zeros_like(pred[:, 7]),  
                                            target_cls=torch.zeros(int(gt_boxes.size(0))))     
                      
        # Comput F1-score
        F1_score = 2*P*R / (P + R + 1e-16)
        F1_scores = np.append(F1_scores, F1_score)
    return F1_scores

def cal_per(F1_scores, threshold):
    return len(F1_scores[F1_scores > threshold])/len(F1_scores)


def main(exps, seq, result_root, start_frame, end_frame, frame_rate, threshold):                               # specify seq
    per_results = []
    F1_results = []
    for exp in exps[0: len(exps) - 1]:
        exp_path = os.path.join(result_root, exp, '{}.txt'.format(seq))   
        base_path = os.path.join(result_root, exps[len(exps)-1], '{}.txt'.format(seq)) 
        exp_data = load_and_bracketing(exp_path, start_frame, end_frame)
        base_data = load_and_bracketing(base_path, start_frame, end_frame)
        F1_scores = eval_F1(exp_data, base_data, frame_rate)
        per = cal_per(F1_scores, threshold)
        F1_results.append(F1_scores)
        per_results.append(per)
    print(per_results)
    best = np.argmax(per_results)
    print('The best one is:', best)

if __name__ == '__main__':
    exps_str =  """
            576_quarter-dla_34_mot17_half_hm
            576_half-dla_34_mot17_half_hm
            576_dla_34_mot17_half_hm
            640_quarter-dla_34_mot17_half_hm
            640_half-dla_34_mot17_half_hm
            640_dla_34_mot17_half_hm
            704_quarter-dla_34_mot17_half_hm
            704_half-dla_34_mot17_half_hm
            704_dla_34_mot17_half_hm
            864_quarter-dla_34_mot17_half_hm
            864_half-dla_34_mot17_half_hm
            864_dla_34_mot17_half_hm        
            1088_quarter-dla_34_mot17_half_hm
            1088_half-dla_34_mot17_half_hm
            1088_dla_34_mot17_half_hm
            """
    seqs_str = '''MOT17-02-SDP
                  MOT17-04-SDP
                  MOT17-05-SDP
                  MOT17-09-SDP
                  MOT17-10-SDP
                  MOT17-11-SDP'''
    result_root = '/u40/xur86/projects/DeepScale/datasets/MOT17/images/results/'
    exps = [exp.strip() for exp in exps_str.split()]
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(exps, 
         seq=seqs[0], 
         result_root=result_root,
         start_frame = 360,
         end_frame = 450,
         frame_rate = 5,
         threshold = 0.7)
