# This script is an extension of track_half_multiknob.py (with frame rate adaptation)
# Author: Renjie Xu
# Time: 2023/4/24

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
import copy
import matplotlib.pyplot as plt
from tracker.multitracker import JDETracker, STrack, joint_stracks, sub_stracks, remove_duplicate_stracks
from tracker import matching
from tracking_utils.kalman_filter import KalmanFilter
from tracker.basetrack import TrackState
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
from tracking_utils.evaluation_multiknob import Evaluator as Evaluator_multiknob
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

def compare_hms(hm, hm_knob):
    det_rate_list = []
    hm = _nms(hm)
    hm_knob = _nms(hm_knob)
    hm = heatmap_to_binary(hm, 0.4)
    hm_knob = heatmap_to_binary(hm_knob, 0.4)
    hm = hm.squeeze()                       
    hm_knob = hm_knob.squeeze(0)
    for i in range(hm_knob.shape[0]):
        det_rate_list.append(torch.div(torch.sum(hadamard_operation(hm_knob[0], hm_knob[i])), torch.sum(hm_knob[0])))
    return det_rate_list

def update_config(det_rate_list, threshold_config):                      # the threshold is step-wise               
    config_fps_sorted = [14, 11, 13, 8, 10, 7, 5, 12, 9, 4, 6, 2, 1, 3, 0]      # the avg fps of the configurations from high to low: averaged by 10 runs
    if threshold_config == 'C0':
        thresholds = [0, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
    if threshold_config == 'C1':
        thresholds = [0, 99, 99, 0.85, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
    if threshold_config == 'C2':
        thresholds = [0, 99, 99, 0.99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
    if threshold_config == 'C3':
        thresholds = [0, 99, 99, 0, 99, 99, 0.90, 99, 99, 99, 99, 99, 99, 99, 99]
    if threshold_config == 'C4':
        thresholds = [0, 99, 99, 0, 99, 99, 0, 99, 99, 0.85, 99, 99, 99, 99, 99]
    if threshold_config == 'C5':
        thresholds = [0, 99, 99, 0, 99, 99, 0, 99, 99, 0, 99, 99, 0.85, 99, 99]
    if threshold_config == 'C6':
        thresholds = [0, 0.99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
    if threshold_config == 'C7':
        thresholds = [0, 99, 0.99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]

    configs_candidates = [idx for idx, det_rate in enumerate(det_rate_list) if det_rate > thresholds[idx]]
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

def convert_results(results):
    converted_results = []
    for frame_id, tlwhs, track_ids in results:
        new_data = []
        for tlwh, track_id in zip(tlwhs, track_ids):
            if track_id < 0:
                continue
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h
            new_entry = (frame_id, track_id, x1, y1, x2, y2, 1, 1, 0)
            new_data.append(new_entry)
        converted_results.extend(new_data)
    return converted_results

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
    
def object_association(seg_start_frame_id, dets_list, id_feature_list, max_time_lost, interval=1):
    history_tracked_stracks = []
    history_lost_stracks = []
    history_removed_stracks = []
    frame_id = 0
    kalman_filter = KalmanFilter()
    results_seg = []
    for i, (dets, id_feature) in enumerate(zip(dets_list, id_feature_list)):        
        if i % interval != 0:
            continue
        frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                            (tlbrs, f) in zip(dets[:, :5], id_feature)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in history_tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, history_lost_stracks)
        # Predict the current location with KF
        #for strack in strack_pool:
            #strack.predict()
        STrack.multi_predict(strack_pool)
        dists = matching.embedding_distance(strack_pool, detections)
        #dists = matching.iou_distance(strack_pool, detections)
        dists = matching.fuse_motion(kalman_filter, dists, strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.4)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < opt.conf_thres:
                continue
            track.activate(kalman_filter, frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        
        for track in history_lost_stracks:
            if frame_id - track.end_frame > max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        history_tracked_stracks = [t for t in history_tracked_stracks if t.state == TrackState.Tracked]
        history_tracked_stracks = joint_stracks(history_tracked_stracks, activated_starcks)
        history_tracked_stracks = joint_stracks(history_tracked_stracks, refind_stracks)
        history_lost_stracks = sub_stracks(history_lost_stracks, history_tracked_stracks)
        history_lost_stracks.extend(lost_stracks)
        history_lost_stracks = sub_stracks(history_lost_stracks, history_removed_stracks)
        history_removed_stracks.extend(removed_stracks)
        history_tracked_stracks, history_lost_stracks = remove_duplicate_stracks(history_tracked_stracks, history_lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in history_tracked_stracks if track.is_activated]
        
        online_tlwhs = []
        online_ids = []        
        for t in output_stracks:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        results_seg.append((seg_start_frame_id + i, online_tlwhs, online_ids))
        if interval != 1:
            for j in range(1, interval):
                results_seg.append((seg_start_frame_id + i + j, online_tlwhs, online_ids))
    return convert_results(results_seg)

def get_best_interval(seg_start_frame_id, interval_list, dets_list, id_feature_list, max_time_lost, threshold=0.9):
    def cal_mota(gt, data):
        acc = []
        evaluator = Evaluator_multiknob(gt, 'mot')
        acc.append(evaluator.eval_file(data))
        metrics = mm.metrics.motchallenge_metrics
        summary = Evaluator_multiknob.get_summary(acc, ['data'], metrics)
        mota = summary.iloc[0]['mota']
        return mota
    
    best_interval_candidates = []
    results_gt = object_association(seg_start_frame_id, dets_list, id_feature_list, max_time_lost, interval_list[0])
    for interval in interval_list[1:]:
        results_seg = object_association(seg_start_frame_id, dets_list, id_feature_list, max_time_lost, interval)
        mota = cal_mota(results_gt, results_seg)
        print('{}: '.format(interval), mota)
        if mota > threshold:
            best_interval_candidates.append(interval)
    best_interval = max(best_interval_candidates, default=1)
    return best_interval

def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30):
    imgsz_list = [(1088, 608), (864, 480), (704, 384), (640, 352), (576, 320)]
    model_list = ['full-dla_34', 'half-dla_34', 'quarter-dla_34']
    interval_list = [1, 2, 3, 6] # fr = 30, 15, 10, 5
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
        if (i - start_frame) % opt.switch_period == 0 or i == start_frame:
            best_config_idx = 0 # res and model
            best_interval = 1   # frame rate
            frame_cnt = 1       # count the frames in segments
            dets_list = []
            id_feature_list = []
        if (i - start_frame) % best_interval != 0:
            frame_id += 1
            continue
        best_config = configs[best_config_idx]
        best_imgsz, best_model = best_config.split('+')
        dataloader.set_image_size(ast.literal_eval(best_imgsz))
        path, img, img0 = dataloader.__getitem__(i)
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        count_config.append(best_config_idx)                                          # count the selected configuration for statistics
        # run tracking
        timer.tic()
        if (i - start_frame) % opt.switch_period == 0 or i == start_frame:
            print('Running switching...')
            online_targets, hm, hm_knob = tracker.update_hm(blob, img0, 'full-dla_34-multiknob')
            det_rate_list = compare_hms(hm, hm_knob)                                  # calculate the detection rate
            best_config_idx = update_config(det_rate_list, opt.threshold_config)      # determine the optimal configuration based on the rule
            seg_start_frame_id = copy.deepcopy(frame_id + 2)
        else:
            online_targets, dets, id_feature, = tracker.update_hm(blob, img0, best_model)
            dets_list.append(dets)
            id_feature_list.append(id_feature)
            frame_cnt += 1

        print('Running imgsz: {} model: {} interval: {} on image: {}'.format(best_imgsz, best_model, best_interval, str(frame_id + 1)))
        online_tlwhs = []
        online_ids = []        
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)

        if frame_cnt == opt.segment:
            print('Selecting the best interval...')
            best_interval = get_best_interval(seg_start_frame_id, interval_list, dets_list, id_feature_list, max_time_lost=int(frame_rate / 30.0 * opt.track_buffer), threshold=0.9)
            print('The best interval is: ', best_interval)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        if best_interval != 1:
            for j in range(1, best_interval):
                results.append((frame_id + 1 + j, online_tlwhs, online_ids))
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
    torch.cuda.set_device(0)
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
