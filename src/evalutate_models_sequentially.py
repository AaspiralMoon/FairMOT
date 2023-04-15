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

exp_id = 'test'           # modify here
result_root = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/results'
exp_path = osp.join(result_root, exp_id)
xlsx_path = osp.join(exp_path, 'summary_{}.xlsx'.format(exp_id))
model_root = '/nfs/u40/xur86/projects/DeepScale/FairMOT/exp/mot/mot17_half_full-dla34_coco_crowdhuman_multires_finetune_model_30'  # modify here

start_point = 2           # modify here
end_point = 20
imgsz_list = [(1088, 608), (864, 480), (704, 384), (640, 352), (576, 320)]
data = {}

for i in range(start_point, end_point + 1):  # model_idx
    mota_list = []
    for j in range(0, 5):                    # resolution
        cmd_str = 'python track_half.py \
                    --exp_id {} \
                    --task mot \
                    --load_model {}/model_{}.pth \
                    --imgsize_index {}'.format(exp_id, model_root, i, j)
        os.system(cmd_str)
        mota = get_mota(xlsx_path)
        mota_list.append(mota)
    data['model_{}'.format(i)] = mota_list

df = pd.DataFrame(data, index=imgsz_list)
df.to_excel(osp.join(exp_path, 'mota_multires_{}_{}.xlsx'.format(start_point, end_point)), engine="openpyxl")