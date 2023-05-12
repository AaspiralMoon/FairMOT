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

class Client:
    def __init__(self, server_address, port, is_client=True):
        self.received_byte = b''
        self.soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        self.is_client = is_client
        self.send_size = 0
        self.receive_size = 0
        try:
            if self.is_client:
                self.connection = self.soc
                self.connection.connect((server_address, port))
                print('Successful Connection to the Server.\n')
            else:            
               self.soc.bind((server_address, port))
               self.soc.listen(1)
               print('Waiting for connections...\n')
               self.connection, client_info = self.soc.accept()
               
        except BaseException as e:
            print('Error Connecting to the Server: {msg}'.format(msg=e))
            self.soc.close()
            print('Socket Closed.\n')
        
    def send(self, data):
        data_byte = pickle.dumps(data)
        size = len(data_byte)    
        self.send_size += size
        self.connection.sendall(struct.pack(">L", size) + data_byte)      # the msg is data_size + data_byte
        
    def receive(self, buffer_size=4096):        
        received_data = None
        payload_size = struct.calcsize(">L")
        while len(self.received_byte) < payload_size:
            self.received_byte += self.connection.recv(buffer_size)                               # first receive the size of image data: msg_size
        packed_msg_size = self.received_byte[:payload_size]
        self.received_byte = self.received_byte[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        self.receive_size += msg_size
        while len(self.received_byte) < msg_size:
            self.received_byte += self.connection.recv(buffer_size)
        frame_data = self.received_byte[:msg_size]                                                # then receive the image data
        self.received_byte = self.received_byte[msg_size:]    
        received_data = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        return received_data

def pre_processing(img0, img_size=(1088, 608), do_letterbox=True, do_transformation=True):
    if do_letterbox and do_transformation:
        img, _, _, _ = letterbox(img0, width=img_size[0], height=img_size[1])
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
    elif do_letterbox and not do_transformation:
        img, _, _, _ = letterbox(img0, width=img_size[0], height=img_size[1])
    elif not do_letterbox and do_transformation:
        img = img0
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
    else:
        raise NotImplementedError
    return img

def main(opt, client, data_root, seqs):
    total_communication_time = 0
    total_client_time = 0
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
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
            start_encoding = time.time()
            _, img0 = cv2.imencode('.jpg', img0, encode_param)        # encoding
            end_encoding = time.time()
            total_client_time += (end_encoding - start_encoding)
            img_info = {'frame_id': int(i + 1), 'img0': img0}
            start_communication = time.time()
            client.send(('original_img', img_info))
            end_communication = time.time()
            total_communication_time += (end_communication - start_communication)
    time_info = {'total_communication_time': total_communication_time, 'total_client_time': total_client_time}
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