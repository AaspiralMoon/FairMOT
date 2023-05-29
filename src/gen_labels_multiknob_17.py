import os.path as osp
import os
import numpy as np

def mkdir_if_missing(d):
    if not osp.exists(d):
        os.makedirs(d)

def xyxy2xywh(x):   # Convert bounding box format from [x1, y1, x2, y2] (bottom left, top right) to [x, y, w, h] (top, left)
    if len(x.shape) == 1:
        y = np.zeros_like(x)
        y[0] = x[0]
        y[1] = x[1]
        y[2] = x[2] - x[0]
        y[3] = x[3] - x[1]
        y[4] = x[4]
    else:
        y = np.zeros_like(x)
        y[:, 0] = x[:, 0]
        y[:, 1] = x[:, 1]
        y[:, 2] = x[:, 2] - x[:, 0]
        y[:, 3] = x[:, 3] - x[:, 1]
        y[:, 4] = x[:, 4]
    return y

seq_root = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17_multiknob/train_yolo'
label_root = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17_multiknob/results_yolo'

seqs = ['MOT17-02-SDP',
        'MOT17-04-SDP',
        'MOT17-05-SDP',
        'MOT17-09-SDP',
        'MOT17-10-SDP',
        'MOT17-11-SDP',
        'MOT17-13-SDP']

imgsize_list = [1088, 864, 704, 640, 576]
model_list = ['full', 'half', 'quarter']

print('Cleaning...')
cmd_str = 'find {} -type f -name "*.txt" -not -path "*/labels_with_ids/*" -delete'.format(seq_root) # delete all existing txt files
os.system(cmd_str)
print('Clean up finished!')

for seq in seqs:
    seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
    seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
    seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

    for imgsz in imgsize_list:
        for m in model_list:
            seq_label_root = osp.join(label_root, seq, '{}_{}'.format(imgsz, m))
            output_path = osp.join(seq_root, seq, '{}_{}'.format(imgsz, m))
            mkdir_if_missing(output_path)
            for fid, f in enumerate(os.listdir(seq_label_root)):
                print('working on {} knob: {}_{} image: {}'.format(seq, imgsz, m, int(fid + 1)))
                bboxes = np.loadtxt(osp.join(seq_label_root, f))
                bboxes = xyxy2xywh(bboxes)
                if bboxes.size == 0:
                    with open(osp.join(output_path, '{:06d}.txt'.format(fid + 1)), 'a') as f:
                            f.write('')
                elif len(bboxes.shape) == 1:
                        x, y, w, h, score = bboxes
                        tid_curr = 0 # doesn't matter
                        if score >= 0.4:
                            x += w / 2
                            y += h / 2
                            label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                                tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
                            with open(osp.join(output_path, '{:06d}.txt'.format(fid + 1)), 'a') as f:
                                f.write(label_str)
                else: 
                    for bbox in bboxes:
                        x, y, w, h, score = bbox
                        tid_curr = 0 # doesn't matter
                        if score < 0.4:
                            break
                        x += w / 2
                        y += h / 2
                        label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                            tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
                        with open(osp.join(output_path, '{:06d}.txt'.format(fid + 1)), 'a') as f:
                            f.write(label_str)