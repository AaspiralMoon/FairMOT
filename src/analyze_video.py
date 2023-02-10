from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import subprocess
import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch
import pickle

from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation2 import Evaluator
import datasets.dataset.jde as datasets

from tracking_utils.utils import mkdir_if_missing
from opts import opts
from track_half import eval_seq

def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,1,0\n'
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


def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30, interval=1):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    for i, (path, img, img0) in enumerate(dataloader):
        if frame_id % interval != 0:
            frame_id += 1
            continue
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        blob = torch.from_numpy(img).cuda().unsqueeze(0)

        online_targets = tracker.update(blob, img0)

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
        if interval != 1:
            for i in range(1, interval):
                results.append((frame_id + 1 + i, online_tlwhs, online_ids))

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

    return frame_id, timer.average_time, timer.calls

def mkdir_if_missing(d):
    if not osp.exists(d):
        os.makedirs(d)

def get_video_length(path):
    cmd_str = 'ffprobe "%s" -show_entries format=duration -of compact=p=0:nk=1 -v 0'%path
    get_time = subprocess.check_output(cmd_str, shell=True)
    timeT = int(float(get_time.strip()))
    return timeT

def video2img(video_path, img_path, frame_rate=30):
    mkdir_if_missing(img_path)
    cmd_str = 'ffmpeg -i {} -r {} -f image2 {}/%06d.jpg'.format(video_path, frame_rate, img_path)
    os.system(cmd_str)


def video_split(video_path, clip_path, start_time=0, interval=10):
    video_length = get_video_length(video_path)
    index=1
    while start_time < video_length and interval <= video_length:
        save_path = osp.join(clip_path, 'clip{}'.format(index))
        mkdir_if_missing(save_path)
        cmd_str = 'ffmpeg -ss %s -i %s -c copy -t %s %s.mp4 -loglevel quiet -y'%(start_time, video_path, interval,'%s/clip%s'%(save_path, index))
        mkdir_if_missing(osp.join(save_path,))
        print(cmd_str)
        returnCmd = subprocess.call(cmd_str, shell=True)
        start_time += interval
        index += 1

def evaluate_mota(gt_filename, result_filename, evaluation_path, seq, exp_name):
    accs = []
    seqs = [seq]
    evaluator = Evaluator(gt_filename, 'mot')
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
    # Evaluator.save_summary(summary, os.path.join(evaluation_path, 'summary_{}.xlsx'.format(exp_name)))
    return summary.iloc[0]['mota']


def tune_knob(opt, img_path, result_path, knob_1, knob_2, knob_3):
    opt = opts().init()
    opt.img_size = knob_1
    opt.arch = knob_2
    opt.load_model = '../models/{}.pth'.format(opt.arch)
    dataloader = datasets.LoadImages(img_path, opt.img_size)
    result_filename = osp.join(result_path, '{}_{}_{}.txt'.format(knob_1, knob_2, knob_3))
    eval_seq(opt, dataloader, 'mot', result_filename,
                          save_dir=None, show_image=False, frame_rate=30, interval=int(30/knob_3))

def get_optimum(list, threshold):
    for i in list:
        if i['mota'] >= threshold:
            optimum = i['configuration']
    return optimum

def main(clip_path, clip_id, list_resolution, list_model, list_frame_rate, threshold):
    img_path = osp.join(clip_path, clip_id, 'img')
    result_path = osp.join(clip_path, clip_id, 'results')
    evaluation_path = osp.join(clip_path, clip_id, 'evaluation')
    mkdir_if_missing(result_path)
    mkdir_if_missing(evaluation_path)
    video2img(osp.join(clip_path, clip_id, '{}.mp4'.format(clip_id)), img_path)      # decode video into frames
    mota_list = []
    for res in list_resolution:
        for m in list_model:
            for fr in list_frame_rate:
                # tune_knob(opt, 
                #           img_path=img_path,
                #           result_path=result_path,
                #           knob_1=res,
                #           knob_2=m,
                #           knob_3=fr)

                mota_score = evaluate_mota(gt_filename=osp.join(result_path, '{}_{}_{}.txt'.format(list_resolution[0], list_model[0], list_frame_rate[0])),
                             result_filename=osp.join(result_path, '{}_{}_{}.txt'.format(res, m, fr)),
                             evaluation_path=evaluation_path, 
                             seq='video', 
                             exp_name='{}_{}_{}'.format(res, m, fr))
                
                mota_dict = {}
                mota_dict['configuration'] = '{}_{}_{}'.format(res, m, fr)
                mota_dict['mota'] = mota_score
                mota_list.append(mota_dict)


    mota_list_sorted = sorted(mota_list, key=lambda x: x['mota'],  reverse=True)
    optimal_configuration = get_optimum(mota_list_sorted, threshold)

    return optimal_configuration

if __name__ == '__main__':
    data_root = '../videos/'
    video_path = osp.join(data_root, 'MOT17-09-SDP-QP50.mp4')
    exp_name = 'MOT17-09-SDP-QP50'
    clip_path = osp.join(data_root, exp_name)
    video_split(video_path, clip_path, start_time=0, interval=1)
    list_resolution = [(1088, 608), (864, 480), (704, 384), (640, 352), (576, 320)]
    list_model = ['dla_34', 'half-dla_34', 'quarter-dla_34']
    list_frame_rate = [30, 15, 10, 5, 1]
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    optimal_configuration_list = {}
    for clip_id in os.listdir(clip_path):
        optimal_configuration_list[clip_id] = main(clip_path=clip_path,
                                                                  clip_id = clip_id,
                                                                  list_resolution=list_resolution,
                                                                  list_model=list_model,
                                                                  list_frame_rate=list_frame_rate,
                                                                  threshold=0.7)
    
    with open(osp.join(clip_path, 'optimal_configuration.txt'),'w') as file:
        for k, v in optimal_configuration_list.items():
            file.write(k + ": " + v + "\n")
     
