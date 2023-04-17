# This script is for automating multiple exps
# Author: Renjie Xu
# Time: 2023/3/26

import os
import time

# track_half_multiknob with different sp and thresh
# sp_list = [40]
# thresh_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11']
# for sp in sp_list:
#     for thresh in thresh_list:
#         cmd_str = 'python track_half_multiknob.py --exp_id multiknob_{}_{}_separate_0.2 --task mot_multiknob \
#                 --load_model ../exp/mot_multiknob/gen_datasets_multiknob_full_crowdhuman/full-dla_34-multiknob.pth \
#                 --load_full_model ../exp/mot_multiknob/gen_datasets_multiknob_full_crowdhuman/full-dla_34.pth \
#                 --load_half_model ../exp/mot_multiknob/gen_datasets_multiknob_full_crowdhuman/half-dla_34.pth \
#                 --load_quarter_model ../exp/mot_multiknob/gen_datasets_multiknob_full_crowdhuman/quarter-dla_34.pth \
#                 --switch_period {} --threshold_config {}'.format(sp, thresh, sp, thresh)
#         os.system(cmd_str)

# track_half with different model and imgsz
# imgsz_list = [(1088, 608), (864, 480), (704, 384), (640, 352), (576, 320)]
# imgsz_idx = [0, 1, 2, 3, 4]
# model_list = ['full-dla_34', 'half-dla_34', 'quarter-dla_34']
qp_list = [i for i in range(35, 46)]
# for idx in imgsz_idx:
#     for m in model_list:
#         cmd_str = 'python track_half.py --exp_id {}_{}_multires --task mot --load_model ../models/{}.pth --imgsize_index {} --arch {}'.format(imgsz_list[idx][0], m[:m.find('-')], m, idx, m)
#         os.system(cmd_str) 

# for idx in imgsz_idx:
#     for m in model_list:
#         cmd_str = 'python track_half.py --exp_id {}_{}_multires_classifier --task mot --load_model ../exp/mot_multiknob/gen_datasets_multiknob/{}.pth --imgsize_index {} --arch {}'.format(imgsz_list[idx][0], m[:m.find('-')], m, idx, m)
#         os.system(cmd_str)

for qp in qp_list:
    cmd_str = 'CUDA_VISIBLE_DEVICES=3 python track_half.py \
                      --exp_id qp_{} \
                      --task mot \
                      --load_model /nfs/u40/xur86/projects/DeepScale/FairMOT/exp/mot/mot17_half_full-dla34_coco_crowdhuman/model_30.pth \
                      --qp {}'.format(qp, qp)
    os.system(cmd_str)
    