# This script is for automating multiple exps
# Author: Renjie Xu
# Time: 2023/3/26

import os
import time
import os.path as osp
import pandas as pd
import numpy as np
import time
import json

def get_mota(xlsx_path):
    df = pd.read_excel(xlsx_path, engine='openpyxl')
    overall_mota = df.loc[df['Unnamed: 0'] == 'OVERALL', 'mota'].values[0]
    return overall_mota

result_root = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/results'

# track_half_multiknob with different sp and thresh
sp_list = [40]
thresh_list = ['C5', 'C7']

for sp in sp_list:
    for thresh in thresh_list:
        avg_fps_dict = {}
        for t in range(1, 21):                              # run exps for several times
            exp_id = 'multiknob_yolo_0.4_{}_{}_verify'.format(sp, thresh)
            exp_path = osp.join(result_root, exp_id)
            result_path = osp.join(exp_path, 'avg_fps.txt')
            cmd_str = 'CUDA_VISIBLE_DEVICES=3 python track_half_multiknob.py \
                    --exp_id {} \
                    --task mot_multiknob \
                    --load_model ../models/full-yolo-multiknob.pth \
                    --load_full_model ../models/full-yolo.pth \
                    --load_half_model ../models/half-yolo.pth \
                    --load_quarter_model ../models/quarter-yolo.pth \
                    --arch full-yolo \
                    --reid_dim 64 \
                    --switch_period {} \
                    --threshold_config {}'.format(exp_id, sp, thresh)
            os.system(cmd_str)
            avg_fps = np.loadtxt(result_path)
            avg_fps_dict['{}'.format(t)] = avg_fps.item()
        average = sum(avg_fps_dict.values()) / len(avg_fps_dict)
        avg_fps_dict['average'] = round(average, 2)
        np.savetxt(result_path, np.asarray([average]), fmt='%.2f')
        with open(result_path.replace('avg_fps.txt', 'avg_fps_list.json'), 'w') as file:
            file.write(json.dumps(avg_fps_dict))

# for sp in sp_list:
#     for thresh in thresh_list:
#         cmd_str = 'CUDA_VISIBLE_DEVICES=2 python track_half_multiknob_fr.py \
#                 --exp_id test2 \
#                 --task mot_multiknob \
#                 --load_model /nfs/u40/xur86/projects/DeepScale/FairMOT/exp/mot_multiknob/multiknob_res_and_model_full_crowdhuman_multires_freeze_real_1.00_1200/model_1101.pth \
#                 --load_full_model ../models/full-dla_34.pth \
#                 --load_half_model ../models/half-dla_34.pth \
#                 --load_quarter_model ../models/quarter-dla_34.pth \
#                 --segment 20 \
#                 --switch_period {} --threshold_config {}'.format(sp, thresh)
#         os.system(cmd_str)

# for sp in sp_list:
#     for thresh in thresh_list:
#         exp_id = 'multiknob_fr_0.4_{}_{}'.format(sp, thresh)
#         exp_path = osp.join(result_root, exp_id)
#         cmd_str = 'CUDA_VISIBLE_DEVICES=1 python track_half_multiknob_fr.py \
#                 --exp_id {} \
#                 --task mot_multiknob \
#                 --load_model /nfs/u40/xur86/projects/DeepScale/FairMOT/exp/mot_multiknob/multiknob_res_and_model_full_crowdhuman_multires_freeze_real_1.00_1200/model_1101.pth \
#                 --load_full_model ../models/full-dla_34.pth \
#                 --load_half_model ../models/half-dla_34.pth \
#                 --load_quarter_model ../models/quarter-dla_34.pth \
#                 --segment 20 \
#                 --switch_period {} --threshold_config {}'.format(exp_id, sp, thresh)
#         start_time = time.time()
#         os.system(cmd_str)
#         end_time = time.time()
#         execution_time = end_time - start_time
#         np.savetxt(osp.join(exp_path, 'overall_execution_time.txt'), np.asarray([execution_time]), fmt='%.1f')

# track_half with different model and imgsz
# imgsz_list = [(1088, 608), (864, 480), (704, 384), (640, 352), (576, 320)]
# imgsz_idx = [0, 1, 2, 3, 4]
# model_list = ['full-dla_34', 'half-dla_34', 'quarter-dla_34']
# interval_list = [1, 2, 3, 6]
# qp_list = [i for i in range(35, 46)]
# for idx in imgsz_idx:
#     for m in model_list:
#         cmd_str = 'python track_half.py --exp_id {}_{}_multires --task mot --load_model ../models/{}.pth --imgsize_index {} --arch {}'.format(imgsz_list[idx][0], m[:m.find('-')], m, idx, m)
#         os.system(cmd_str) 

# for idx in imgsz_idx:
#     for m in model_list:
#         cmd_str = 'python track_half.py --exp_id {}_{}_multires_classifier --task mot --load_model ../exp/mot_multiknob/gen_datasets_multiknob/{}.pth --imgsize_index {} --arch {}'.format(imgsz_list[idx][0], m[:m.find('-')], m, idx, m)
#         os.system(cmd_str)

# for qp in qp_list:
#     cmd_str = 'CUDA_VISIBLE_DEVICES=3 python track_half.py \
#                       --exp_id qp_{} \
#                       --task mot \
#                       --load_model /nfs/u40/xur86/projects/DeepScale/FairMOT/exp/mot/mot17_half_full-dla34_coco_crowdhuman/model_30.pth \
#                       --gen_dets \
#                       --qp {}'.format(qp, qp)
#     os.system(cmd_str)

# for interval in interval_list:
#         for idx in imgsz_idx:
#                 for m in model_list:
#                         cmd_str = 'CUDA_VISIBLE_DEVICES=2 python track_half.py \
#                                 --exp_id mot17_half_{}_{}_multires_interval_{} \
#                                 --task mot \
#                                 --load_model ../models/{}.pth \
#                                 --arch {} \
#                                 --imgsize_index {} \
#                                 --interval {}'.format(imgsz_list[idx][0], m[:m.find('-')], interval, m, m, idx, interval)
#                         os.system(cmd_str)

# for interval in interval_list:
#         for idx in imgsz_idx:
#                 for m in model_list:
#                         cmd_str = 'CUDA_VISIBLE_DEVICES=3 python track_half.py \
#                                 --exp_id mot17_half_{}_{}_interval_{} \
#                                 --task mot \
#                                 --load_model ../models/baselines/{}.pth \
#                                 --arch {} \
#                                 --imgsize_index {} \
#                                 --interval {}'.format(imgsz_list[idx][0], m[:m.find('-')], interval, m, m, idx, interval)
#                         os.system(cmd_str)

# for m in model_list:
#         for idx in imgsz_idx:
#                 for interval in interval_list:
#                         result_path = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/results/'
#                         excel_path = osp.join(result_path, 'mot17_half_{}_{}_interval_{}'.format(imgsz_list[idx][0], m[:m.find('-')], interval), 'summary2_mot17_half_{}_{}_interval_{}.xlsx'.format(imgsz_list[idx][0], m[:m.find('-')], interval))
#                         mota = get_mota(excel_path)
#                         print('{}_{}_{}'.format(imgsz_list[idx][0], m[:m.find('-')], interval), mota)
                        
                
                