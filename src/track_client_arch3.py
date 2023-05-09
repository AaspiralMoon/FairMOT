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
import motmetrics as mm
import logging

from tracker.multitracker import JDETracker
import datasets.dataset.jde as datasets
from opts import opts
from track_client_arch1 import Client, pre_processing
from track_half import write_results
from tracking_utils.evaluation import Evaluator
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
from tracking_utils.timer import Timer

def main(opt, client, data_root, seqs):
    logger.setLevel(logging.INFO)
    accs = []
    result_root = os.path.join(data_root, '..', 'results', opt.exp_id)
    mkdir_if_missing(result_root)
    for seq in seqs:
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        tracker = JDETracker(opt, frame_rate=frame_rate)
        start_frame = int(len(dataloader) / 2)
        dataset_info = {'seq': seq, 'frame_rate': frame_rate}
        client.send(('dataset_info', dataset_info))
        results = []
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        timer_client_computation = Timer()
        for i, (path, img, img0) in enumerate(dataloader):
            if i < start_frame:
                continue
            if (i - start_frame) % opt.switch_period == 0:
                img_info = {'frame_id': int(i + 1), 'img0': img0}
                client.send(('original_img', img_info))
                received_data = client.receive()
                if received_data:
                    best_imgsz = received_data['best_imgsz']
                    best_model = received_data['best_model']
                    dets = received_data['dets']
                    id_feature = received_data['id_feature']
                    online_targets = tracker.object_association(dets, id_feature)
            else:
                img = pre_processing(img0, best_imgsz)
                blob = torch.from_numpy(img).cuda().unsqueeze(0)
                online_targets, _, _ = tracker.update_hm(blob, img0, best_model)
                print('Running imgsz: {} model: {} on image: {}'.format(best_imgsz, best_model, str(i + 1)))
            online_tlwhs = [t.tlwh for t in online_targets if t.tlwh[2] * t.tlwh[3] > opt.min_box_area and t.tlwh[2] / t.tlwh[3] <= 1.6]
            online_ids = [t.track_id for t in online_targets if t.tlwh[2] * t.tlwh[3] > opt.min_box_area and t.tlwh[2] / t.tlwh[3] <= 1.6]
            results.append((str(i + 1), online_tlwhs, online_ids))
        write_results(result_filename, results, data_type='mot')
        evaluator = Evaluator(data_root, seq, data_type='mot')
        accs.append(evaluator.eval_file(result_filename))   
    
    # get summary
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