# This script is for tracking on the server
# Author: Renjie Xu
# Time: 2023/5/6
# Command: CUDA_VISIBLE_DEVICES=3 python track_server_arch1.py --exp_id XX --task mot_multiknob --load_model ../models/full-yolo-multiknob.pth --load_full_model ../models/full-yolo.pth --load_half_model ../models/half-yolo.pth --load_quarter_model ../models/quarter-yolo.pth --arch full-yolo --reid_dim 64 --switch_period 40 --threshold_config C1

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
import time
import json

from tracker.multitracker import JDETracker
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
    model_list = ['full', 'half', 'quarter']
    configs = []
    for imgsz in imgsz_list:
        for m in model_list:
            configs.append('{}+{}'.format(imgsz, m))

    result_root = os.path.join(data_root, '..', 'results', opt.exp_id)
    mkdir_if_missing(result_root)

    accs = []
    total_server_time = 0
    num_frames = 0
    total_data_size = 0

    while True:
        received_data, msg_size = server.receive()
        if received_data:
            data_type, data = received_data

            if data_type == 'dataset_info':
                dataset_info = data
                seq = dataset_info['seq']
                img0_width = dataset_info['img0_width']
                img0_height = dataset_info['img0_height']
                frame_rate = dataset_info['frame_rate']
                start_frame = dataset_info['start_frame']
                last_frame = dataset_info['last_frame']
                results = []
                result_filename = os.path.join(result_root, '{}.txt'.format(seq))
                tracker = JDETracker(opt, frame_rate=frame_rate)
                continue

            elif data_type == 'full_img':
                img_info = data
                frame_id = img_info['frame_id']
                img = img_info['img']
                total_data_size += msg_size

            elif data_type == 'terminate':
                time_info = data

                total_communication_time = time_info['total_communication_time']
                total_client_time = time_info['total_client_time']
                
                avg_communication_time = round(total_communication_time * 1000 / num_frames, 1)
                avg_client_time = round(total_client_time * 1000 / num_frames, 1)
                avg_server_time = round(total_server_time * 1000 / num_frames, 1)
                avg_fps = round(num_frames / (total_communication_time + total_client_time + total_server_time), 1)
                avg_network_traffic = round(total_data_size / (num_frames * 1024), 1)

                avg_time_info = {'avg_fps': avg_fps, 'avg_server_time': avg_server_time, 'avg_client_time': avg_client_time, 'avg_communication_time': avg_communication_time, 'avg_network_traffic': avg_network_traffic}
                with open(osp.join(result_root, 'avg_time_info.json'), 'w') as file:
                    file.write(json.dumps(avg_time_info))

                # evaluate MOTA
                metrics = mm.metrics.motchallenge_metrics
                mh = mm.metrics.create()
                summary = Evaluator.get_summary(accs, seqs, metrics)
                strsummary = mm.io.render_summary(
                    summary,
                    formatters=mh.formatters,
                    namemap=mm.io.motchallenge_metric_names
                )
                print(strsummary)
                print('num_frames: ', num_frames)
                Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(opt.exp_id)))
                break

            else:
                print('Unknown data type: {}'.format(data_type))
                continue
                
            if frame_id is not None and img is not None:
                if (frame_id - 1 - start_frame) % opt.switch_period == 0:
                    best_config_idx = 0
                best_config = configs[best_config_idx]
                best_imgsz, best_model = best_config.split('+')

                start_server_decoding = time.time()
                img = cv2.imdecode(img, 1)
                end_server_decoding = time.time()
                total_server_time += (end_server_decoding - start_server_decoding)
                img = pre_processing(img0, ast.literal_eval(best_imgsz))              
                blob = torch.from_numpy(img).cuda().unsqueeze(0)
                start_server_computation = time.time()                  # start time for server computation
                if (frame_id - 1 - start_frame) % opt.switch_period == 0:
                    print('Running switching...')
                    online_targets, hm_knob = tracker.update_hm(blob, img0, 'full-multiknob')
                    det_rate_list = compare_hms(hm_knob)
                    best_config_idx = update_config(det_rate_list, opt.threshold_config)
                else:
                    online_targets = tracker.update_hm(blob, img0, best_model)
                end_server_computation = time.time()                   # end time for server computation
                total_server_time += (end_server_computation - start_server_computation)
                num_frames += 1
                print('Running imgsz: {} model: {} on image: {}'.format(best_imgsz, best_model, str(frame_id)))
                online_tlwhs = [t.tlwh for t in online_targets if t.tlwh[2] * t.tlwh[3] > opt.min_box_area and t.tlwh[2] / t.tlwh[3] <= 1.6]
                online_ids = [t.track_id for t in online_targets if t.tlwh[2] * t.tlwh[3] > opt.min_box_area and t.tlwh[2] / t.tlwh[3] <= 1.6]
                results.append((frame_id, online_tlwhs, online_ids))

            if frame_id == last_frame:
                write_results(result_filename, results, data_type='mot')
                evaluator = Evaluator(data_root, seq, data_type='mot')
                accs.append(evaluator.eval_file(result_filename))

if __name__ == "__main__":
    server = Client(server_address='130.113.68.165', port=8223, is_client=False)
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
