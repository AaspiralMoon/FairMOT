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
import torch
import logging
import cv2

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
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    for seq in seqs:
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        tracker = JDETracker(opt, frame_rate=frame_rate)
        start_frame = int(len(dataloader) / 2)
        dataset_info = {'seq': seq, 'frame_rate': frame_rate}
        client.send(('dataset_info', dataset_info))
        results = []
        for i, (path, img, img0) in enumerate(dataloader):
            if i < start_frame:
                continue
            num_frames += 1
            if (i - start_frame) % opt.switch_period == 0:
                start_encoding = time.time()
                _, img0 = cv2.imencode('.jpg', img0, encode_param)        # encoding
                end_encoding = time.time()
                total_client_time += (end_encoding - start_encoding)
                img_info = {'frame_id': int(i + 1), 'img0': img0}
                start_communication = time.time()
                client.send(('original_img', img_info))
                end_communication = time.time()
                total_communication_time += (end_communication - start_communication)
                received_data, _ = client.receive()
                if received_data:
                    best_imgsz = received_data['best_imgsz']
                    best_model = received_data['best_model']
                    dets = received_data['dets']
                    id_feature = received_data['id_feature']
                    start_client_computation = time.time()
                    online_targets = tracker.object_association(dets, id_feature)
                    end_client_computation = time.time()
                    total_client_time += (end_client_computation - start_client_computation)
            else:
                img = pre_processing(img0, best_imgsz)
                blob = torch.from_numpy(img).cuda().unsqueeze(0)
                start_client_computation = time.time()
                online_targets, _, _ = tracker.update_hm(blob, img0, best_model)
                end_client_computation = time.time()
                total_client_time += (end_client_computation - start_client_computation)
                print('Running imgsz: {} model: {} on image: {}'.format(best_imgsz, best_model, str(i + 1)))
            online_tlwhs = [t.tlwh for t in online_targets if t.tlwh[2] * t.tlwh[3] > opt.min_box_area and t.tlwh[2] / t.tlwh[3] <= 1.6]
            online_ids = [t.track_id for t in online_targets if t.tlwh[2] * t.tlwh[3] > opt.min_box_area and t.tlwh[2] / t.tlwh[3] <= 1.6]
            results.append((str(i + 1), online_tlwhs, online_ids))
        results_info = {'results': results}
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