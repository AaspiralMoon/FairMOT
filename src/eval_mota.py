from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import motmetrics as mm
from tracking_utils.evaluation2 import Evaluator

seqs = ['MOT17-02-SDP',
        'MOT17-04-SDP',
        'MOT17-05-SDP',
        'MOT17-09-SDP',
        'MOT17-10-SDP',
        'MOT17-11-SDP',
        'MOT17-13-SDP']
accs = []
for seq in seqs: 
    gt_filename = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/results/test_original_gt/{}.txt'.format(seq)
    result_filename = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/results/test_blanked/{}.txt'.format(seq)
    evaluator = Evaluator(gt_filename, 'mot')
    accs.append(evaluator.eval_file(result_filename))

metrics = mm.metrics.motchallenge_metrics
mh = mm.metrics.create()
summary = Evaluator.get_summary(accs, seqs, metrics)
strsummary = mm.io.render_summary(
    summary,
    formatters=mh.formatters,
    namemap=mm.io.motchallenge_metric_names
)
print(strsummary)