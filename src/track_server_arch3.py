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
import logging
import motmetrics as mm
import torch
import ast

from tracker.multitracker import JDETracker
from tracking_utils.log import logger
from tracking_utils.timer import Timer

from opts import opts
from track_client_arch1 import Client, pre_processing
from track_half_multiknob import compare_hms, update_config

def main(opt, server, data_root, seqs):
    logger.setLevel(logging.INFO)
    imgsz_list = [(1088, 608), (864, 480), (704, 384), (640, 352), (576, 320)]
    model_list = ['full-dla_34', 'half-dla_34', 'quarter-dla_34']
    configs = []
    for imgsz in imgsz_list:
        for m in model_list:
            configs.append('{}+{}'.format(imgsz, m))

    current_seq = None
    tracker = None
    seq = None
    frame_rate = None
    frame_id = None
    img0 = None

    while True:
        received_data = server.receive()
        if received_data:
            data_type, data = received_data

            if data_type == 'dataset_info':
                dataset_info = data
                seq = dataset_info['seq']
                frame_rate = dataset_info['frame_rate']

            elif data_type == 'original_img':
                img_info = data
                frame_id = img_info['frame_id']
                img0 = img_info['img0']

            elif data_type == 'terminate':
                break

            else:
                print('Unknown data type: {}'.format(data_type))
                continue

            if seq is not None and frame_rate is not None:
                if seq != current_seq:
                    current_seq = seq
                    frame_id = None
                    img0 = None
                    tracker = JDETracker(opt, frame_rate=frame_rate)
                    
            if frame_id is not None and img0 is not None:         
                img = pre_processing(img0)         
                blob = torch.from_numpy(img).cuda().unsqueeze(0)
                print('Running switching...')
                hm_knob, dets, id_feature = tracker.update_hm(blob, img0, 'full-dla_34-multiknob', if_object_association=False)
                det_rate_list = compare_hms(hm_knob)                                  # calculate the detection rate
                best_config_idx = update_config(det_rate_list, opt.threshold_config)
                best_config = configs[best_config_idx]
                best_imgsz, best_model = best_config.split('+')
                print('Running imgsz: (1088, 608) model: full-dla_34 on image: {}'.format(str(frame_id)))
                best_config_info = {'best_imgsz': ast.literal_eval(best_imgsz), 'best_model': best_model, 'dets': dets, 'id_feature': id_feature}
                server.send(best_config_info)

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
