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

os.environ['CV_IO_MAX_IMAGE_PIXELS']='1099511627776'

class Client:
    def __init__(self, server_address,port, is_client=True):
        self.received_byte = b""
        self.soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        self.is_client = is_client
        self.send_size = 0
        self.receive_size = 0
        try:
            if self.is_client:
                self.connection = self.soc
                self.connection.connect((server_address, port))
                print("Successful Connection to the Server.\n")
            else:            
               self.soc.bind((server_address, port))
               self.soc.listen(1)
               print('Waiting for connections...')
               self.connection, client_info = self.soc.accept()
               
        except BaseException as e:
            print("Error Connecting to the Server: {msg}".format(msg=e))
            self.soc.close()
            print("Socket Closed.")
        
    def send(self, data):
        data_byte = pickle.dumps(data)
        size = len(data_byte)    
        self.send_size += size
        self.connection.sendall(struct.pack(">L", size) + data_byte)
        
    def receive(self, buffer_size=4096):        
        received_data = None
        payload_size = struct.calcsize(">L")
        while len(self.received_byte) < payload_size:
            self.received_byte += self.connection.recv(buffer_size)
        packed_msg_size = self.received_byte[:payload_size]
        self.received_byte = self.received_byte[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        self.receive_size += msg_size
        while len(self.received_byte) < msg_size:
            self.received_byte += self.connection.recv(buffer_size)
        frame_data = self.received_byte[:msg_size]
        self.received_byte = self.received_byte[msg_size:]    
        received_data = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        return received_data

def pre_processing(img0, img_size=(1088, 608)):
    img, _, _, _ = letterbox(img0, width=img_size[0], height=img_size[1])
    # img = img[:, :, ::-1].transpose(2, 0, 1)
    # img = np.ascontiguousarray(img, dtype=np.float32)
    # img /= 255.0
    return img

def main(opt, client, data_root, seqs):
    for seq in seqs:
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
        start_frame = int(len(dataloader) / 2)
        for i, (path, img, img0) in enumerate(dataloader):
            if i < start_frame:
                continue
            start_pre_processing = time.time()
            img2 = pre_processing(img0)
            end_pre_processing = time.time()
            print('Pre-processing time: ', end_pre_processing - start_pre_processing)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            result, img3 = cv2.imencode('.jpg', img2, encode_param)
            start_client_send = time.time()
            client.send(img3)
            end_client_send = time.time()
            print('Client sending time: ', end_client_send - start_client_send)
            import sys
            sys.exit(0)

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