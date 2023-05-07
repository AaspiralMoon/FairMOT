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

from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets

from tracking_utils.utils import mkdir_if_missing
from opts import opts
from track_client import Client, letterbox
from track_half import write_results

def pre_processing(img0, img_size=(1088, 608)):
    img, _, _, _ = letterbox(img0, width=img_size[0], height=img_size[1])
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0
    return img

def main(opt, server, data_root, seqs):
    result_root = os.path.join(data_root, '..', 'results', opt.exp_id)
    mkdir_if_missing(result_root)

    current_seq = None
    tracker = None
    seq = None
    frame_rate = None
    last_frame_id = None
    frame_id = None
    img0 = None
    accs = []

    while True:
        received_data = server.receive()
        if received_data:
            data_type, data = received_data

            if data_type == 'D':
                dataset_info = data
                seq = dataset_info['seq_id']
                frame_rate = dataset_info['frame_rate']
                last_frame_id = dataset_info['last_frame_id']

            elif data_type == 'I':
                img_info = data
                frame_id = img_info['frame_id']
                img0 = img_info['img0']

            elif data_type == 'T':
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

            if seq is not None and frame_rate is not None and last_frame_id is not None:
                if seq != current_seq:
                    current_seq = seq
                    results = []
                    frame_id = None
                    img0 = None
                    tracker = JDETracker(opt, frame_rate=frame_rate)
                    result_filename = os.path.join(result_root, '{}.txt'.format(seq))
                    
            if frame_id is not None and img0 is not None:
                img = pre_processing(img0)
                blob = torch.from_numpy(img).cuda().unsqueeze(0)
                online_targets = tracker.update(blob, img0)

                online_tlwhs = [t.tlwh for t in online_targets if t.tlwh[2] * t.tlwh[3] > opt.min_box_area and t.tlwh[2] / t.tlwh[3] <= 1.6]
                online_ids = [t.track_id for t in online_targets if t.tlwh[2] * t.tlwh[3] > opt.min_box_area and t.tlwh[2] / t.tlwh[3] <= 1.6]

                results.append((frame_id, online_tlwhs, online_ids))

            if frame_id == last_frame_id:
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
