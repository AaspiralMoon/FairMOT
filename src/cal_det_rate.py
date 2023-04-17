# This script is for calculating the detection rate of different QPs
# Author: Renjie Xu
# Time: 2023/4/16

import os
import os.path as osp
import matplotlib.pyplot as plt

def count_det(file_path):                 # count the detection numbers, i.e., the number of lines in the txt
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return len(lines)


result_root = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/results'
exp_id_list = []

seqs = ['MOT17-02-SDP',
        'MOT17-04-SDP',
        'MOT17-05-SDP',
        'MOT17-09-SDP',
        'MOT17-10-SDP',
        'MOT17-11-SDP',
        'MOT17-13-SDP']

det_dict = {}
for exp_id in exp_id_list:
    exp_det_dict = {}
    exp_path = osp.join(result_root, exp_id)
    for seq in seqs:
        seq_det_list = []
        seq_path = osp.join(exp_path, seq)
        det_path = osp.join(seq_path, '{}_dets'.format(seq))
        for file in os.listdir(det_path):
            num_det = count_det(file)
            seq_det_list.append(num_det)
        exp_det_dict['{}'.format(seq)] = seq_det_list
    det_dict['{}'.format(exp_id)] = exp_det_dict


det_rate_dict = {}
qp_0 = det_dict['{}'.format(exp_id_list[0])]
for exp_id in exp_id_list[1:]:
    exp_det_rate_dict = {}
    qp = det_dict['{}'.format(exp_id)]
    for seq in seqs:
        qp_0_seq = qp_0['{}'.format(seq)]
        qp_seq = qp['{}'.format(seq)]
        seq_det_rate_list = [x / y for x, y in zip(qp_seq, qp_0_seq)]
        exp_det_rate_dict['{}'.format(seq)] = seq_det_rate_list
    det_dict['{}'.format(exp_id)] = exp_det_rate_dict

# Create a 3x2 grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 10))
axes = axes.flatten()  # Flatten the axes array for easy indexing

# Define colors for each line
colors = ['red', 'blue', 'green', 'orange']

# Plot the line plots for each dataset
for i, ax in enumerate(axes):
    if i <= len(seqs):
        for j, exp_id in enumerate(exp_id_list[1:]):
            ax.plot(list(range(len(det_dict[exp_id][seqs[i]]) + 1, 2 * len(det_dict[exp_id][seqs[i]]) + 1)), det_dict[exp_id][seqs[i]], label='{}'.format(exp_id), color=colors[j])
        ax.set_title('{}'.format(seqs[i]))
        ax.set_xlabel('Frames')
        ax.set_ylabel('Detection Rate')
        ax.legend()

# Remove the last (empty) subplot
fig.delaxes(axes[-1])

# Adjust the layout
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# Save the plot as an image file
plt.savefig('detection rate.png', dpi=300)