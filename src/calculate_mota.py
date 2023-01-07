from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import motmetrics as mm
import numpy as np


from tracking_utils.evaluation import Evaluator

def main(data_root, result_filename, result_root, seq, exp_name, data_type='mot'):
    accs = []
    seqs = [seq]
    evaluator = Evaluator(data_root, seq, data_type)
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
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))




if __name__ == '__main__':
    data_root = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/train'
    result_filename = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/results/full_1088_30/MOT17-02-SDP.txt'
    result_root = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/results/test'
    seq = 'MOT17-02-SDP'
    exp_name = 'test'
    data_type = 'mot'
    main(data_root=data_root,
         result_filename=result_filename,
         result_root=result_root,
         seq=seq,
         exp_name='test',
         data_type=data_type)
