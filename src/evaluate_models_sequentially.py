# This script is for evaluating models sequentially and recording their results
# Author: Renjie Xu
# Time: 2023/4/15

import pandas as pd
import os
import os.path as osp

def get_mota(xlsx_path):
    df = pd.read_excel(xlsx_path, engine='openpyxl')
    overall_mota = df.loc[df['Unnamed: 0'] == 'OVERALL', 'mota'].values[0]
    return overall_mota

# Multi-Res
# exp_id = 'test'           # modify here
# result_root = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/results'
# exp_path = osp.join(result_root, exp_id)
# xlsx_path = osp.join(exp_path, 'summary_{}.xlsx'.format(exp_id))
# model_root = '/nfs/u40/xur86/projects/DeepScale/FairMOT/exp/mot_multiknob/multiknob_yolo_freeze_1200'  # modify here

# start_point = 10          # modify here
# end_point = 10
# imgsz_list = [(1088, 608), (864, 480), (704, 384), (640, 352), (576, 320)]
# data = {}

# for i in range(start_point, end_point + 1):  # model_idx
#     mota_list = []
#     for j in range(0, 5):                    # resolution
#         cmd_str = 'CUDA_VISIBLE_DEVICES=1 python track_half.py \
#                     --exp_id {} \
#                     --task mot_multiknob \
#                     --load_model {}/model_{}.pth \
#                     --arch full-yolo \
#                     --reid_dim 64 \
#                     --imgsize_index {}'.format(exp_id, model_root, i, j)       # modify here
#         os.system(cmd_str)
#         mota = get_mota(xlsx_path)
#         mota_list.append(mota)
#     data['model_{}'.format(i)] = mota_list
#     df = pd.DataFrame(data, index=imgsz_list)
#     df.to_excel(osp.join(exp_path, 'mota_multires_{}_to_{}.xlsx'.format(start_point, end_point)), engine="openpyxl")

# df = pd.DataFrame(data, index=imgsz_list)
# df.to_excel(osp.join(exp_path, 'mota_multires_{}_to_{}.xlsx'.format(start_point, end_point)), engine="openpyxl")

# # Single-Res
# exp_id = 'full-yolo_coco_15_to_100'           # modify here
# result_root = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/results'
# exp_path = osp.join(result_root, exp_id)
# xlsx_path = osp.join(exp_path, 'summary_{}.xlsx'.format(exp_id))
# model_root = '/nfs/u40/xur86/projects/DeepScale/FairMOT/exp/mot/mot17_half_full-yolo_coco'  # modify here

# start_point = 15             # modify here
# end_point = 100
# imgsz_list = [(1088, 608)]
# data = {}

# for i in range(start_point, end_point + 1):  # model_idx
#     mota_list = []
#     cmd_str = 'CUDA_VISIBLE_DEVICES=2 python track_half.py \
#                 --exp_id {} \
#                 --task mot \
#                 --load_model {}/model_{}.pth \
#                 --arch full-yolo \
#                 --reid_dim 64'.format(exp_id, model_root, i)       # modify here
#     os.system(cmd_str)
#     mota = get_mota(xlsx_path)
#     mota_list.append(mota)
#     data['model_{}'.format(i)] = mota_list
#     df = pd.DataFrame(data, index=imgsz_list)
#     df.to_excel(osp.join(exp_path, 'mota_multires_{}_to_{}.xlsx'.format(start_point, end_point)), engine="openpyxl")

# df = pd.DataFrame(data, index=imgsz_list)
# df.to_excel(osp.join(exp_path, 'mota_multires_{}_to_{}.xlsx'.format(start_point, end_point)), engine="openpyxl")

# Multi-Knob
exp_id = 'multiknob_yolo_freeze_1200_1199_to_1199'           # modify here
result_root = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/results'
exp_path = osp.join(result_root, exp_id)
xlsx_path = osp.join(exp_path, 'summary_{}.xlsx'.format(exp_id))
model_root = '/nfs/u40/xur86/projects/DeepScale/FairMOT/exp/mot_multiknob/multiknob_yolo_freeze_1200'  # modify here

start_point = 1199           # modify here
end_point = 1199
thresh_list = ['C1']
data = {}

for i in range(start_point, end_point + 1):  # model_idx
    mota_list = []
    for thresh in thresh_list:
        cmd_str = 'CUDA_VISIBLE_DEVICES=3 python track_half_multiknob.py \
                --exp_id {} \
                --task mot_multiknob \
                --load_model {}/model_{}.pth \
                --load_full_model ../models/full-yolo.pth \
                --load_half_model ../models/half-yolo.pth \
                --load_quarter_model ../models/quarter-yolo.pth \
                --arch full-yolo \
                --reid_dim 64 \
                --switch_period 40 \
                --threshold_config {}'.format(exp_id, model_root, i, thresh)
        os.system(cmd_str)
        mota = get_mota(xlsx_path)
        mota_list.append(mota)
    data['model_{}'.format(i)] = mota_list
    df = pd.DataFrame(data, index=thresh_list)
    df.to_excel(osp.join(exp_path, 'mota_multiknob_{}_to_{}.xlsx'.format(start_point, end_point)), engine="openpyxl")

df = pd.DataFrame(data, index=thresh_list)
df.to_excel(osp.join(exp_path, 'mota_multiknob_{}_to_{}.xlsx'.format(start_point, end_point)), engine="openpyxl")