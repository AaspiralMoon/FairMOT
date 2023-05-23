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
import matplotlib.pyplot as plt
from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets
from models.decode import _nms
from tracking_utils.utils import mkdir_if_missing
from opts import opts

# python track_half_multiknob.py --task mot_multiknob --load_model ../models/full-dla_34.pth --load_half_model ../exp/mot/mot17_half_half-dla34_with_pretrain/model_last.pth --load_quarter_model ../models/quarter-dla_34.pth --switch_period 30

def heatmap_to_binary(heatmap, threshold):
    binary = (heatmap > threshold).to(torch.float32)
    return binary

def hadamard_operation(A, B): # Element-wise Hadamard product
    return A * B 

def compare_hms(hm_knob):
    det_rate_list = []
    hm_knob = _nms(hm_knob)
    hm_knob = heatmap_to_binary(hm_knob, 0.4)                  
    hm_knob = hm_knob.squeeze(0)
    for i in range(hm_knob.shape[0]):
        det_rate_list.append(torch.div(torch.sum(hadamard_operation(hm_knob[0], hm_knob[i])), torch.sum(hm_knob[0])))
    return det_rate_list

def update_config(det_rate_list, threshold_config):                      # the threshold is step-wise               
    config_fps_sorted = [14, 11, 8, 13, 10, 7, 5, 12, 4, 9, 2, 6, 1, 3, 0]  
    if threshold_config == 'C1':
        thresholds = [99, 99, 99, 0.99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
    if threshold_config == 'C2':
        thresholds = [99, 0.99, 99, 0, 99, 99, 0.95, 99, 99, 0.99, 99, 99, 99, 99, 99]
    if threshold_config == 'C3':
        thresholds = [99, 99, 99, 99, 0.95, 99, 0, 99, 99, 0.85, 99, 99, 0.95, 99, 99]
    if threshold_config == 'C4':
        thresholds = [99, 99, 99, 99, 0.90, 0.90, 99, 0.90, 0.90, 0, 99, 99, 0.85, 99, 99]
    if threshold_config == 'C5':
        thresholds = [99, 99, 99, 99, 99, 0.80, 99, 0.80, 0.90, 99, 0.90, 99, 0, 99, 99]
    if threshold_config == 'C6':
        thresholds = [99, 99, 99, 99, 99, 99, 99, 99, 0, 99, 99, 0.65, 99, 99, 99]
    if threshold_config == 'C7':
        thresholds = [99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 0, 99, 99, 0.60]
    configs_candidates = [idx for idx, det_rate in enumerate(det_rate_list) if det_rate >= thresholds[idx]]
    if len(configs_candidates) == 0:          # if no config satisfies the requirement, return the golden config
        best_config_idx = 0
    else:
        best_config_idx = min((config_fps_sorted.index(candidates), candidates) for candidates in configs_candidates)[1]
    return best_config_idx

def plot_config_distribution(result_root, count_config, seq=None):
    count_dict = {}
    configs = []
    imgsz_list = [1088, 864, 704, 640, 576]
    model_list = ['full', 'half', 'quarter']
    for imgsz in imgsz_list:
        for m in model_list:
            configs.append('{}_{}'.format(imgsz, m))
    for config_idx in count_config:
        if configs[config_idx] in count_dict:
            count_dict[configs[config_idx]] += 1
        else:
            count_dict[configs[config_idx]] = 1
    # plot and save the pie chart
    labels = list(count_dict.keys())
    sizes = list(count_dict.values())
    colors = plt.cm.tab20(np.linspace(0, 1, len(configs)))
    config_colors = {config: colors[i] for i, config in enumerate(configs)}
    plt.rcParams.update({'font.size': 12})
    wedges, texts, autotexts = plt.pie(
        sizes, labels=labels, colors=[config_colors[label] for label in labels], autopct="%.1f%%", startangle=90, textprops=dict(color='w')
    )
    plt.axis('equal')
    plt.legend(wedges, labels, title='Configurations', loc='lower right', bbox_to_anchor=(1.1, 0), fontsize=10)
    if seq is None:
        plt.title('Selected Configurations in ALL', fontsize=14)
        plt.savefig(osp.join(result_root, 'config_distribution_all.png'), dpi=300, bbox_inches='tight')
    else:
        plt.title('Selected Configurations in Seq{}'.format(seq), fontsize=14)
        plt.savefig(osp.join(result_root, 'config_distribution_{}.png'.format(seq)), dpi=300, bbox_inches='tight')
    plt.close()
    plt.clf()
            
def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
        # save_format = '{frame},{id},{x1},{y1},{w},{h},1,1,0\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))

def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30):
    imgsz_list = [(1088, 608), (864, 480), (704, 384), (640, 352), (576, 320)]
    model_list = ['full-dla_34', 'half-dla_34', 'quarter-dla_34']
    configs = []
    for imgsz in imgsz_list:
        for m in model_list:
            configs.append('{}+{}'.format(imgsz, m))
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    count_config = []
    len_all = len(dataloader)
    start_frame = int(len_all / 2)
    frame_id = int(len_all / 2)
    best_config_idx = 0
    for i, (path, img, img0) in enumerate(dataloader):
        if i < start_frame:
            continue
        if (i - start_frame) % opt.switch_period == 0:
            best_config_idx = 0
        best_config = configs[best_config_idx]
        best_imgsz, best_model = best_config.split('+')
        dataloader.set_image_size(ast.literal_eval(best_imgsz))
        path, img, img0 = dataloader.__getitem__(i)
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        count_config.append(best_config_idx)                                          # count the selected configuration for statistics
        # run tracking
        timer.tic()
        if (i - start_frame) % opt.switch_period == 0:
            print('Running switching...')
            online_targets, hm_knob = tracker.update_hm(blob, img0, 'full-dla_34-multiknob')
            det_rate_list = compare_hms(hm_knob)                                  # calculate the detection rate
            best_config_idx = update_config(det_rate_list, opt.threshold_config)      # determine the optimal configuration based on the rule
        else:
            online_targets, _, _ = tracker.update_hm(blob, img0, best_model)

        print('Running imgsz: {} model: {} on image: {}'.format(best_imgsz, best_model, str(frame_id + 1)))
        online_tlwhs = []
        online_ids = []        
        #online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                #online_scores.append(t.score)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        #results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    # save results
    write_results(result_filename, results, data_type)
    #write_results_score(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls, count_config


def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    count_config_seqs = []
    for seq in seqs:
        output_dir = os.path.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))

        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        nf, ta, tc, count_config = eval_seq(opt, dataloader, data_type, result_filename,
                                    save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        count_config_seqs += count_config                                                           # count the selected configurations over all seqs
        plot_config_distribution(result_root, count_config, seq.split("-")[1])                      # plot and save the selected configuration distribution in each seq

        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    if opt.is_profiling:
        np.savetxt(osp.join(result_root, 'avg_fps.txt'), 1.0 / np.asarray([avg_time]), fmt='%.2f')   # save the profile (average fps)
    plot_config_distribution(result_root, count_config_seqs)                                         # plot and save the selected configuration distribution over all seqs
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

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
    opt = opts().init()
    if not opt.val_mot16:
        seqs_str = '''KITTI-13
                      KITTI-17
                      ADL-Rundle-6
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte'''
        #seqs_str = '''TUD-Campus'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    else:
        seqs_str = '''MOT16-02
                      MOT16-04
                      MOT16-05
                      MOT16-09
                      MOT16-10
                      MOT16-11
                      MOT16-13'''
        data_root = os.path.join(opt.data_dir, 'MOT16/train')
    if opt.test_mot16:
        seqs_str = '''MOT16-01
                      MOT16-03
                      MOT16-06
                      MOT16-07
                      MOT16-08
                      MOT16-12
                      MOT16-14'''
        seqs_str = '''MOT16-06 MOT16-07 MOT16-08'''
        data_root = os.path.join(opt.data_dir, 'MOT16/test')
    if opt.test_mot15:
        seqs_str = '''ADL-Rundle-1
                      ADL-Rundle-3
                      AVG-TownCentre
                      ETH-Crossing
                      ETH-Jelmoli
                      ETH-Linthescher
                      KITTI-16
                      KITTI-19
                      PETS09-S2L2
                      TUD-Crossing
                      Venice-1'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/test')
    if opt.test_mot17:
        seqs_str = '''MOT17-01-SDP
                      MOT17-03-SDP
                      MOT17-06-SDP
                      MOT17-07-SDP
                      MOT17-08-SDP
                      MOT17-12-SDP
                      MOT17-14-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/test')
    if opt.val_mot17:
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP'''
        # seqs_str = '''MOT17-02-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/train')
    if opt.val_mot15:
        seqs_str = '''Venice-2
                      KITTI-13
                      KITTI-17
                      ETH-Bahnhof
                      ETH-Sunnyday
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte
                      ADL-Rundle-6
                      ADL-Rundle-8
                      ETH-Pedcross2
                      TUD-Stadtmitte'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    if opt.val_mot20:
        seqs_str = '''MOT20-01
                      MOT20-02
                      MOT20-03
                      MOT20-05
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/train')
    if opt.test_mot20:
        seqs_str = '''MOT20-04
                      MOT20-06
                      MOT20-07
                      MOT20-08
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/test')
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name=opt.exp_id,
         show_image=False,
         save_images=False,
         save_videos=False)
