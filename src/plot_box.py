import cv2
import numpy as np
import numpy as np

def nms_gt(boxes, overlapThresh=0.5):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

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


def nms(dets, thresh):
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

def save_image_with_boxes(img, boxes, output_path, color=(0, 255, 0), thickness=2):
    # If there's only one box, ensure it is a 2D array
    if len(boxes.shape) == 1:
        boxes = boxes.reshape(1, -1)

    # Draw each box
    for box in boxes:
        x1, y1, x2, y2 = box[:4]  # Ignore the confidence value
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

    # Save the image
    cv2.imwrite(output_path, img)

# Example usage:
seq_id = 'MOT17-02-SDP'
img_id = '000600'
img_path = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/train/{}/img1/{}.jpg'.format(seq_id, img_id)
box_path = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/results/576_quarter_dla_dets/{}_dets/{}.txt'.format(seq_id, img_id)
gt_box_path = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/results/MOT17_gt/{}/{}.txt'.format(seq_id, img_id)

boxes = np.loadtxt(box_path)
boxes_after_nms = nms(boxes, thresh=0.5)
gt_boxes = np.loadtxt(gt_box_path)
gt_boxes_after_nms = nms_gt(gt_boxes)

img = cv2.imread(img_path)
save_image_with_boxes(img.copy(), boxes, 'detected_boxes_quarter.jpg')
save_image_with_boxes(img.copy(), boxes_after_nms, 'detected_boxes_quarter_nms.jpg')
save_image_with_boxes(img.copy(), gt_boxes, 'gt_boxes.jpg')
save_image_with_boxes(img.copy(), gt_boxes_after_nms, 'gt_boxes_nms.jpg')