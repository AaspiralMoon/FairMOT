# This script is for tracking on the client
# Author: Renjie Xu
# Time: 2023/5/6
# Command: python track_client.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import socket
import pickle
import struct
import os
import os.path as osp
import numpy as np
import time
import cv2

import datasets.dataset.jde as datasets
from opts import opts
from datasets.dataset.jde import letterbox
from track_client_arch1 import Client, pre_processing

def main(opt, client, data_root, seqs):
    for seq in seqs:
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        start_frame = int(len(dataloader) / 2)
        dataset_info = {'seq': seq, 'frame_rate': frame_rate, 'start_frame': start_frame, 'last_frame': len(dataloader)}
        client.send(('dataset_info', dataset_info))
        for i, (path, img, img0) in enumerate(dataloader):
            if i < start_frame:
                continue
            if (i - start_frame) % opt.switch_period == 0:
                img_info = {'frame_id': int(i + 1), 'img0': img0}
                client.send(('original_img', img_info))
                received_data = client.receive()
                if received_data:
                    best_imgsz = received_data['best_imgsz']
            else:
                img = pre_processing(img0, best_imgsz)
                img_info = {'frame_id': int(i + 1), 'img': img}
                client.send(('scaled_img', img_info))
    client.send(('terminate', None))                     # transmission completed, terminate the connetction

if __name__ == '__main__':
    client = Client(server_address='localhost', port=8223)
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