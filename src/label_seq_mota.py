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

from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets

from tracking_utils.utils import mkdir_if_missing
from opts import opts

def str2float(a):
    return float(a)

def gen_trainset(seq_path, save_path, seq): 
    # init 
    file = []
    data = []
    index = []
    data_dir = os.path.join(save_path, 'train')

    # read file
    with open(seq_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = list(map(str2float, line.strip().split(',')))
            file.append(line)
    file.sort()

    # data transform
    for line in file:
        data_dict = {}
        data_dict['frame_id'] = line[0]
        data_dict['obj_id'] = line[1]
        data_dict['x'] = line[2]
        data_dict['y'] = line[3]
        data_dict['w'] = line[4]
        data_dict['h'] = line[5]
        data.append(data_dict)
        index.append(line[0])
    index_sort = list(set(index))
    index_sort.sort()

    # start processing
    for idx in index_sort:
        area_sum = 0
        speed_sum = 0
        data_dict_train = {}
        data_dict_train['frame_id'] = idx
        data_dict_train['obj_num'] = index.count(idx)
        for i in data:
            if i['frame_id'] == idx:         # calculate average size of objects
                area = i['w']*i['h']
                area_sum = area_sum + area
        for i in data:                       # calculate average speed of objects
            if i['frame_id'] == idx - 1:
                for j in data:
                    if j['frame_id'] == idx:
                        if (i['frame_id'] == j['frame_id'] - 1) and (i['obj_id'] == j['obj_id']):
                            speed_sum =  speed_sum + math.sqrt((j['x'] + j['w']/2 - i['x'] - i['w']/2)**2 + (j['y'] - j['h']/2 - i['y'] + i['h']/2)**2)
        data_dict_train['obj_size'] = area_sum/data_dict_train['obj_num']
        data_dict_train['obj_speed'] = speed_sum/data_dict_train['obj_num']
        line = '{frame_id},{obj_num},{obj_size},{obj_speed}\n'
        line = line.format(frame_id=int(data_dict_train['frame_id']), obj_num=int(data_dict_train['obj_num']), obj_size=data_dict_train['obj_size'], obj_speed=data_dict_train['obj_speed'], )
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        with open(os.path.join(data_dir, '{}_train.txt'.format(seq)), 'a+') as f:
            f.write(line)
    return(data_dir)

def txtseg(seq_path, save_path, seq):                   # divide the result txt into multiple sub-txts based on frame_id
    with open(seq_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            frame_id = int(line[0: 4].strip(','))
            data_dir = os.path.join(save_path, 'segs')
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            with open(os.path.join(data_dir, '{}.txt'.format(frame_id)), 'a+') as temp:
                temp.write(line)
        return data_dir

def label_mota(data_dir, seq, motas):                 # label each frame with mota score  \
    filename = os.path.join(data_dir, '{}_train.txt'.format(seq))
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line, mota in zip(lines, motas):
            line = str(mota) + ',' + line
            with open(os.path.join(data_dir, '{}_labeled.txt'.format(seq)), 'a+') as temp:
                temp.write(line)

def main(data_root, result_root, seqs):
    for seq in seqs:
        txts = []
        motas = []
        save_path = os.path.join(result_root, '{}'.format(seq))
        seq_path = os.path.join(result_root, '{}.txt'.format(seq))
        data_dir_train = gen_trainset(seq_path, save_path, seq)
        data_dir_seg = txtseg(seq_path, save_path, seq)
        temps = os.listdir(data_dir_seg)
        for temp in temps:
            txt_id = temp[0: 4].strip('.')
            txts.append(txt_id)
        for txt in txts: 
            accs = []
            txt_path = os.path.join(data_dir_seg, '{}.txt'.format(txt))
            data_type = 'mot'
            evaluator = Evaluator(data_root, seq, data_type)
            accs.append(evaluator.eval_file(txt_path))

            # get summary
            metrics = mm.metrics.motchallenge_metrics
            mh = mm.metrics.create()
            summary = Evaluator.get_summary(accs, [txt], metrics)
            strsummary = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names
            )
            motas.append(float(summary.loc[['{}'.format(txt)]]['mota']))
            print(strsummary)
        label_mota(data_dir_train, seq, motas)


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
    main(data_root=data_root,
         result_root=result_root,
         seqs=seqs)