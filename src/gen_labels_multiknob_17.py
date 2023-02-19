import os.path as osp
import os
import numpy as np

def mkdir_if_missing(d):
    if not osp.exists(d):
        os.makedirs(d)

seq_root = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17_multiknob/train'
label_root = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17_multiknob/results'

seqs = [s for s in os.listdir(seq_root)]

imgsize_list = [1088, 864, 704, 640, 576]
model_list = ['full', 'half', 'quarter']
qp_list = [10, 20, 30, 40, 50]

# cmd_str = 'find {} -name "*.txt" -type f -delete'.format(seq_root) # delete all existing txt files
# os.system(cmd_str)

for seq in seqs:
    seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
    seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
    seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

    for imgsize in imgsize_list:
        for m in model_list:
            for qp in qp_list:
                seq_label_root = osp.join(label_root, seq, '{}_{}_{}'.format(imgsize, m, qp))
                output_path = osp.join(seq_root, seq, '{}_{}_{}'.format(imgsize, m, qp))
                mkdir_if_missing(output_path)
                for fid, f in enumerate(os.listdir(seq_label_root)):
                    print('working on {} knob: {}_{}_{} image: {}'.format(seq, imgsize, m, qp, int(fid + 1)))
                    bboxes = np.loadtxt(osp.join(seq_label_root, f))
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
                            # if score < 0.4:
                            #     break
                            x += w / 2
                            y += h / 2
                            label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                                tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
                            with open(osp.join(output_path, '{:06d}.txt'.format(fid + 1)), 'a') as f:
                                f.write(label_str)
