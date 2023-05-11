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
    while True:
        received_data = server.receive()
        start_decoding_time = time.time()
        img = cv2.imdecode(received_data, 1)
        end_decoding_time = time.time()
        print('Decoding time: ', end_decoding_time - start_decoding_time)

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
