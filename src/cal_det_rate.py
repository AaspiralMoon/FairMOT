# This script is for calculating the detection rate of different QPs
# Author: Renjie Xu
# Time: 2023/4/16
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def nms_gt(boxes, overlapThresh=0.5):
    if boxes.size == 0: 
        return boxes
    elif len(boxes.shape) == 1: # single box
        return boxes

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index value to the list of picked indexes, then initialize the suppression list (i.e. the list of indexes that will be deleted) using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        # loop over all indexes in the indexes list
        for pos in range(0, last):
            # grab the current index
            j = idxs[pos]

            # find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # compute the ratio of overlap between the computed bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]

            # if there is sufficient overlap, suppress the current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)

        # delete all indexes from the index list that are in the suppression list
        idxs = np.delete(idxs, suppress)

    # return only the bounding boxes that were picked
    return boxes[pick]


def nms(dets, thresh=0.5):
    if dets.size == 0: 
        return dets
    elif len(dets.shape) == 1: # single box
        return dets
    
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return dets[keep]

def calculate_iou(box1, box2):
    # Determine the coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Compute the area of intersection rectangle
    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Compute the area of each bounding box
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Compute the Intersection over Union (IoU)
    iou = inter_area / float(box1_area + box2_area - inter_area)

    return iou

def compare_detections(boxes, gt_boxes, iou_threshold=0.5):
    if boxes.size == 0:                              # check if file is empty
        return 0                                # return 0 detections and 0 true positives if file is empty

    if len(boxes.shape) == 1:  # If boxes contains only one detection, reshape it.
        boxes = boxes.reshape(1, -1)

    num_true_positives = 0
    for gt in gt_boxes:
        best_iou = 0
        for det in boxes:
            iou = calculate_iou(det[:4], gt)
            if iou > best_iou:
                best_iou = iou

        if best_iou > iou_threshold:
            num_true_positives += 1

    return num_true_positives


# def count_det(file_path):                 # count the detection numbers, i.e., the number of lines in the txt
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#     return len(lines)

result_root = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/results'
gt_path = osp.join(result_root, 'MOT17_gt')
imgsz_list = [1088, 864, 704, 640, 576]
# imgsz_list = [1088]
# model_list = ['full', 'half', 'quarter']
model_list = ['full']
exp_id_list = []
for imgsz in imgsz_list:
    for m in model_list:
        exp_id_list.append('{}_{}_dla_dets'.format(imgsz, m))

seqs = ['MOT17-02-SDP',
        'MOT17-04-SDP',
        'MOT17-05-SDP',
        'MOT17-09-SDP',
        'MOT17-10-SDP',
        'MOT17-11-SDP',
        'MOT17-13-SDP']

det_dict = {}
gt_dict = {}
for exp_id in exp_id_list:
    exp_det_dict = {}
    exp_path = osp.join(result_root, exp_id)
    for seq in seqs:
        seq_det_list = []
        seq_gt_list = []
        det_path = osp.join(exp_path, '{}_dets'.format(seq))
        for file in os.listdir(det_path):
            boxes_path = osp.join(det_path, file)
            gt_boxes_path = boxes_path.replace(det_path, gt_path).replace(file, '{}/{}'.format(seq, file))
            boxes = np.loadtxt(boxes_path)
            gt_boxes = np.loadtxt(gt_boxes_path)
            boxes = nms(boxes)
            # gt_boxes = nms_gt(gt_boxes)
            num_dets = compare_detections(boxes, gt_boxes, 0.5)
            seq_det_list.append(num_dets)
            seq_gt_list.append(len(gt_boxes))
        exp_det_dict['{}'.format(seq)] = seq_det_list
        gt_dict['{}'.format(seq)] = seq_gt_list
    det_dict['{}'.format(exp_id)] = exp_det_dict

det_rate_dict = {}
for exp_id in exp_id_list[0:]:
    exp_det_rate_dict = {}
    det = det_dict['{}'.format(exp_id)]
    for seq in seqs:
        det_gt_seq = gt_dict['{}'.format(seq)]
        det_seq = det['{}'.format(seq)]
        seq_det_rate_list = [x / y for x, y in zip(det_seq, det_gt_seq)]
        exp_det_rate_dict['{}'.format(seq)] = seq_det_rate_list
    det_rate_dict['{}'.format(exp_id)] = exp_det_rate_dict


# Select the sequence of interest
seq_name = 'MOT17-13-SDP'

# Extract data for the sequence
seq_data = {exp_id: det_rate_dict[exp_id][seq_name] for exp_id in exp_id_list}

# Convert dictionary to DataFrame
df = pd.DataFrame(seq_data)

# Save DataFrame to CSV
df.to_csv('seq_det_rate.csv', index=False)




# # Create a 3x2 grid of subplots
# fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(30, 40))
# axes = axes.flatten()  # Flatten the axes array for easy indexing

# # Define colors for each line
# colors = ['red', 'blue', 'green', 'orange', 'purple']

# # Plot the line plots for each dataset
# for i, ax in enumerate(axes):
#     if i < len(seqs):
#         for j, exp_id in enumerate(exp_id_list[0:]):
#             ax.plot(list(range(len(det_rate_dict[exp_id][seqs[i]]) + 1, 2 * len(det_rate_dict[exp_id][seqs[i]]) + 1)), det_rate_dict[exp_id][seqs[i]], label='{}'.format(exp_id), color=colors[j])
#         ax.set_title('{}'.format(seqs[i]))
#         ax.set_xlabel('Frames')
#         ax.set_ylabel('Detection Rate')
#         ax.legend()


# # Adjust the layout
# fig.tight_layout()
# fig.subplots_adjust(top=0.9)

# # Save the plot as an image file
# plt.savefig('detection rate.png', dpi=300)