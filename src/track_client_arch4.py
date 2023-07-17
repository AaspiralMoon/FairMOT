# This script is for tracking on the client
# Author: Renjie Xu
# Time: 2023/5/6
# Command: python track_client.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import time
import cv2
import torch
import logging

from tracking_utils.log import logger
from tracker.multitracker import JDETracker
import datasets.dataset.jde as datasets
from opts import opts
from track_client_arch1 import Client, pre_processing

def main(opt, client, data_root, seqs):
    logger.setLevel(logging.INFO)
    total_communication_time = 0
    total_client_time = 0
    num_frames = 0
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    for seq in seqs:
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        img0_width = int(meta_info[meta_info.find('imWidth=') + 8:meta_info.find('\nimHeight')])
        img0_height = int(meta_info[meta_info.find('imHeight=') + 9:meta_info.find('\nimExt')])
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        tracker = JDETracker(opt, frame_rate=frame_rate)
        start_frame = int(len(dataloader) / 2)
        dataset_info = {'seq': seq, 'img0_width': img0_width, 'img0_height': img0_height, 'frame_rate': frame_rate}
        client.send(('dataset_info', dataset_info))
        client_results = []
        id_stracks = None
        tracked_stracks = None
        lost_stracks = None
        removed_stracks = None
        for i, (path, img, img0) in enumerate(dataloader):
            if i < start_frame:
                continue
            num_frames += 1
            if (i - start_frame) % opt.switch_period == 0:
                img = pre_processing(img0, do_letterbox=True, do_transformation=False)                                        # full resolution
                start_client_encoding = time.time()
                _, img = cv2.imencode('.jpg', img, encode_param)        # encoding
                end_client_encoding = time.time()
                total_client_time += (end_client_encoding - start_client_encoding)
                if id_stracks is None:
                    img_info = {'frame_id': int(i + 1), 'img': img, 'task': 'multiknob'}
                else:
                    img_info = {'frame_id': int(i + 1), 'img': img, 'task': 'multiknob', 'id_stracks': id_stracks, 'tracked_stracks': tracked_stracks, 'lost_stracks': lost_stracks, 'removed_stracks': removed_stracks}
                start_communication = time.time()
                client.send(('img', img_info))
                end_communication = time.time()
                total_communication_time += (end_communication - start_communication)
                received_data, _ = client.receive()
                if received_data:
                    best_imgsz = received_data['best_imgsz']
                    best_model = received_data['best_model']
                    do_transfer = received_data['do_transfer']
                has_received_history_info = False
            elif do_transfer:
                img = pre_processing(img0, best_imgsz, do_letterbox=True, do_transformation=False)
                start_client_encoding = time.time()
                _, img = cv2.imencode('.jpg', img, encode_param)         # encoding
                end_client_encoding = time.time()
                total_client_time += (end_client_encoding - start_client_encoding)
                img_info = {'frame_id': int(i + 1), 'img': img, 'task': 'regular'}
                start_communication = time.time()
                client.send(('transfer_img', img_info))
                end_communication = time.time()
                total_communication_time += (end_communication - start_communication)
            else:
                if not has_received_history_info:
                    client.send(('require_stracks', True))
                    received_data, _ = client.receive()
                    if received_data:
                        id_stracks = received_data.get('id_stracks')
                        tracked_stracks = received_data.get('tracked_stracks')
                        lost_stracks = received_data.get('lost_stracks')
                        removed_stracks = received_data.get('removed_stracks')
                        if id_stracks is not None:
                            tracker.frame_id = id_stracks
                            tracker.tracked_stracks = tracked_stracks
                            tracker.lost_stracks = lost_stracks
                            tracker.removed_stracks = removed_stracks
                    has_received_history_info = True
                img = pre_processing(img0, best_imgsz)
                blob = torch.from_numpy(img).cuda().unsqueeze(0)
                start_client_computation = time.time()
                online_targets = tracker.update_hm(blob, img0, best_model)
                end_client_computation = time.time()
                total_client_time += (end_client_computation - start_client_computation)
                print('Running imgsz: {} model: {} on image: {}'.format(best_imgsz, best_model, str(i + 1)))
                online_tlwhs = [t.tlwh for t in online_targets if t.tlwh[2] * t.tlwh[3] > opt.min_box_area and t.tlwh[2] / t.tlwh[3] <= 1.6]
                online_ids = [t.track_id for t in online_targets if t.tlwh[2] * t.tlwh[3] > opt.min_box_area and t.tlwh[2] / t.tlwh[3] <= 1.6]
                client_results.append((str(i + 1), online_tlwhs, online_ids))
                id_stracks = tracker.frame_id
                tracked_stracks = tracker.tracked_stracks
                lost_stracks= tracker.lost_stracks
                removed_stracks = tracker.removed_stracks
        results_info = {'client_results': client_results}
        client.send(('results_info', results_info))
    
    time_info = {'total_communication_time': total_communication_time, 'total_client_time': total_client_time, 'num_frames': num_frames}
    client.send(('terminate', time_info))                     # transmission completed, terminate the connetction

if __name__ == '__main__':
    client = Client(server_address='130.113.68.165', port=8223)
    opt = opts().init()
    seqs_str = '''MOT17-02-SDP
                  MOT17-04-SDP
                  MOT17-05-SDP
                  MOT17-09-SDP
                  MOT17-10-SDP
                  MOT17-11-SDP
                  MOT17-13-SDP'''
    data_root = os.path.join(opt.data_dir, 'MOT17/images/train')
    seqs = [seq.strip() for seq in seqs_str.split()]
    main(opt,
         client=client,
         data_root=data_root,
         seqs=seqs)