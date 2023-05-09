# This script is for tracking on the server
# Author: Renjie Xu
# Time: 2023/5/6
# Command: python track_server.py

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
import ast

from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets

from tracking_utils.utils import mkdir_if_missing
from opts import opts
from track_client_arch1 import Client, pre_processing
from track_half import write_results
from track_half_multiknob import compare_hms, update_config

def main(opt, server, data_root, seqs):
    logger.setLevel(logging.INFO)
    imgsz_list = [(1088, 608), (864, 480), (704, 384), (640, 352), (576, 320)]
    model_list = ['full-dla_34', 'half-dla_34', 'quarter-dla_34']
    configs = []
    for imgsz in imgsz_list:
        for m in model_list:
            configs.append('{}+{}'.format(imgsz, m))

    result_root = os.path.join(data_root, '..', 'results', opt.exp_id)
    mkdir_if_missing(result_root)

    current_seq = None
    tracker = None
    seq = None
    frame_rate = None
    start_frame = None
    last_frame = None
    frame_id = None
    img0 = None
    accs = []

    while True:
        received_data = server.receive()
        if received_data:
            data_type, data = received_data

            if data_type == 'dataset_info':
                dataset_info = data
                seq = dataset_info['seq']
                frame_rate = dataset_info['frame_rate']
                start_frame = dataset_info['start_frame']
                last_frame = dataset_info['last_frame']

            elif data_type == 'original_img':
                img_info = data
                frame_id = img_info['frame_id']
                img0 = img_info['img0']

            elif data_type == 'scaled_img':
                img_info = data
                frame_id = img_info['frame_id']
                img = img_info['img']

            elif data_type == 'terminate':
                metrics = mm.metrics.motchallenge_metrics
                mh = mm.metrics.create()
                summary = Evaluator.get_summary(accs, seqs, metrics)
                strsummary = mm.io.render_summary(
                    summary,
                    formatters=mh.formatters,
                    namemap=mm.io.motchallenge_metric_names
                )
                print(strsummary)
                Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(opt.exp_id)))
                break

            else:
                print('Unknown data type: {}'.format(data_type))
                continue

            if seq is not None and frame_rate is not None and start_frame is not None and last_frame is not None:
                if seq != current_seq:
                    current_seq = seq
                    results = []
                    frame_id = None
                    img0 = None
                    img = None
                    tracker = JDETracker(opt, frame_rate=frame_rate)
                    result_filename = os.path.join(result_root, '{}.txt'.format(seq))
                    
            if (frame_id is not None and img0 is not None) or (frame_id is not None and img is not None):           
                if (frame_id - 1 - start_frame) % opt.switch_period == 0:
                    img = pre_processing(img0)         
                    blob = torch.from_numpy(img).cuda().unsqueeze(0)
                    print('Running switching...')
                    online_targets, hm_knob = tracker.update_hm(blob, img0, 'full-dla_34-multiknob')
                    det_rate_list = compare_hms(hm_knob)                                  # calculate the detection rate
                    best_config_idx = update_config(det_rate_list, opt.threshold_config)
                    best_config = configs[best_config_idx]
                    best_imgsz, best_model = best_config.split('+')
                    print('Running imgsz: (1088, 608) model: full-dla_34 on image: {}'.format(str(frame_id)))
                    best_config_info = {'best_imgsz': ast.literal_eval(best_imgsz), 'best_model': best_model}
                    server.send(best_config_info)
                else:
                    blob = torch.from_numpy(img).cuda().unsqueeze(0)
                    online_targets, _, _ = tracker.update_hm(blob, img0, best_model)
                    print('Running imgsz: {} model: {} on image: {}'.format(best_imgsz, best_model, str(frame_id)))

                online_tlwhs = [t.tlwh for t in online_targets if t.tlwh[2] * t.tlwh[3] > opt.min_box_area and t.tlwh[2] / t.tlwh[3] <= 1.6]
                online_ids = [t.track_id for t in online_targets if t.tlwh[2] * t.tlwh[3] > opt.min_box_area and t.tlwh[2] / t.tlwh[3] <= 1.6]
                results.append((frame_id, online_tlwhs, online_ids))

            if frame_id == last_frame:
                write_results(result_filename, results, data_type='mot')
                evaluator = Evaluator(data_root, seq, data_type='mot')
                accs.append(evaluator.eval_file(result_filename))

if __name__ == "__main__":
    server = Client(server_address='localhost', port=8223, is_client=False)
    opt = opts().init()
    seqs_str = '''MOT17-02-SDP
                  MOT17-04-SDP
                  MOT17-05-SDP
                  MOT17-09-SDP
                  MOT17-10-SDP
                  MOT17-11-SDP
                  MOT17-13-SDP'''
    seqs = [seq.strip() for seq in seqs_str.split()]
    data_root = os.path.join(opt.data_dir, 'MOT17/images/train')
    main(opt,
         server=server,
         data_root=data_root,
         seqs=seqs)
