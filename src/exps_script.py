# This script is for automating multiple exps
# Author: Renjie Xu
# Time: 2023/3/26

import os
import time

# track_half_multiknob with different sp and thresh
# sp_list = [40]
# thresh_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
# for sp in sp_list:
#     for thresh in thresh_list:
#         cmd_str = 'python track_half_multiknob.py --exp_id profiling_multiknob_{}_{} --task mot_multiknob \
#                --load_model ../models/full-dla_34-multiknob.pth --load_half_model ../models/half-dla_34.pth --load_quarter_model ../models/quarter-dla_34.pth \
#                --switch_period {} --threshold_config {}'.format(sp, thresh, sp, thresh)
#         os.system(cmd_str)
#         time.sleep(5)

# track_half with different model and imgsz
imgsz_list = [(1088, 608), (864, 480), (704, 384), (640, 352), (576, 320)]
imgsz_idx = [0, 1, 2, 3, 4]
model_list = ['full-dla_34', 'half-dla_34', 'quarter-dla_34']
# for idx in imgsz_idx:
#     for m in model_list:
#         cmd_str = 'python track_half.py --exp_id {}_{}_multires --task mot --load_model ../models/{}.pth --imgsize_index {} --arch {}'.format(imgsz_list[idx][0], m[:m.find('-')], m, idx, m)
#         os.system(cmd_str) 

for idx in imgsz_idx:
    for m in model_list:
        cmd_str = 'python track_half.py --exp_id {}_{}_without_multires --task mot --load_model ../exp/mot/full_half_quarter_without_multires/{}.pth --imgsize_index {} --arch {}'.format(imgsz_list[idx][0], m[:m.find('-')], m, idx, m)
        os.system(cmd_str)