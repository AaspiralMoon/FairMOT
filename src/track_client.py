# This script is for tracking on the client
# Author: Renjie Xu
# Time: 2023/5/6
# Command: python track_client.py --data_root /nfs/u40/nalaiek/data/MOT17/MOT17/images/train

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

class Client:
    connection = None
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
            # self.soc.close()
            print("Socket Closed.")
        
    def send(self, data):
        data_byte = pickle.dumps(data)
        size = len(data_byte)    
        self.send_size += size
        self.connection.sendall(struct.pack(">L", size) + data_byte)
        
    def receive(self, buffer_size=4096):        
        received_data = None
        try:
            payload_size = struct.calcsize(">L")
            while len(self.received_byte) < payload_size:
                # print("Recv: {}".format(len(data)))
                self.received_byte += self.connection.recv(buffer_size)
            packed_msg_size = self.received_byte[:payload_size]
            self.received_byte = self.received_byte[payload_size:]
            msg_size = struct.unpack(">L", packed_msg_size)[0]
            # print("receive size: {}".format(msg_size))
            self.receive_size += msg_size
            while len(self.received_byte) < msg_size:
                self.received_byte += self.connection.recv(buffer_size)
            frame_data = self.received_byte[:msg_size]
            self.received_byte = self.received_byte[msg_size:]    
            received_data = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        except BaseException as e:
                print("Error Receiving Data from the Client: {msg}.\n".format(msg=e))
                import sys
                sys.exit(0)
                return received_data
        return received_data
    
    
def reshape_image(img0,sizes,best_res_label):
    img, _, _, _ = letterbox(img0, height= sizes[best_res_label][1],
                             width= sizes[best_res_label][0])
    return img, img0.shape

    
def letterbox(img, height=608, width=1088,
              color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh

def main(opt, client, data_root, seqs):
    for seq in seqs:
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        dataset_info = {'seq_id': seq, 'frame_rate': frame_rate, 'last_frame_id': len(dataloader)}
        client.send(('D', dataset_info))
        for i, (path, img, img0) in enumerate(dataloader):
            if i < int(len(dataloader) / 2):
                continue
            img_info = {'frame_id': int(i + 1), 'img0': img0}
            client.send(('I', img_info))
        client.send(('T', None))             # transmission completed, terminate the connetction

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