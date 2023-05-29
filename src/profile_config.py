import os
import os.path as osp
import numpy as np
import json

result_root = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/results'

imgsz_idx = [0, 1, 2, 3, 4]
imgsz_list = [(1088, 608), (864, 480), (704, 384), (640, 352), (576, 320)]

# # DLA
# model_list = ['full-dla_34', 'half-dla_34', 'quarter-dla_34']
# for idx in imgsz_idx:
#     for m in model_list:
#         avg_fps_dict = {}
#         for t in range(1, 11):
#             cmd_str = 'CUDA_VISIBLE_DEVICES=3 python track_half.py --task mot --exp_id profile_{}_{} --load_model ../models/{}.pth --arch {} --imgsize_index {}'.format(imgsz_list[idx][0], m[:m.find('-')], m, m, idx)
#             os.system(cmd_str)
#             result_path = osp.join(result_root, 'profile_{}_{}'.format(imgsz_list[idx][0], m[:m.find('-')]), 'avg_fps.txt')
#             avg_fps = np.loadtxt(result_path)
#             avg_fps_dict['{}'.format(t)] = avg_fps.item()
#         average = sum(avg_fps_dict.values()) / len(avg_fps_dict)
#         avg_fps_dict['average'] = average
#         np.savetxt(result_path, np.asarray([average]), fmt='%.2f')                                          # overwrite to save the average of 3 runs
#         with open(result_path.replace('avg_fps.txt', 'avg_fps_list.json'), 'w') as file:  # also save the results of the 3 runs
#             file.write(json.dumps(avg_fps_dict)) # use `json.loads` to do the reverse

# Yolo
model_list = ['full-yolo', 'half-yolo', 'quarter-yolo']
# for idx in imgsz_idx:
#     for m in model_list:
#         avg_fps_dict = {}
#         for t in range(1, 11):
#             exp_id = 'profile_yolo_{}_{}'.format(imgsz_list[idx][0], m[:m.find('-')])
#             cmd_str = 'CUDA_VISIBLE_DEVICES=3 python track_half.py \
#                         --task mot \
#                         --exp_id {} \
#                         --load_model ../models/{}.pth \
#                         --arch {} \
#                         --reid_dim 64 \
#                         --imgsize_index {}'.format(exp_id, m, m, idx)
#             os.system(cmd_str)
#             result_path = osp.join(result_root, exp_id, 'avg_fps.txt')
#             avg_fps = np.loadtxt(result_path)
#             avg_fps_dict['{}'.format(t)] = avg_fps.item()
#         average = sum(avg_fps_dict.values()) / len(avg_fps_dict)
#         avg_fps_dict['average'] = average
#         np.savetxt(result_path, np.asarray([average]), fmt='%.2f')                                          # overwrite to save the average of 3 runs
#         with open(result_path.replace('avg_fps.txt', 'avg_fps_list.json'), 'w') as file:  # also save the results of the 3 runs
#             file.write(json.dumps(avg_fps_dict)) # use `json.loads` to do the reverse

avg_fps_dict = {}
cnt = 0
for idx in imgsz_idx:
    for m in model_list:
        exp_id = 'profile_yolo_{}_{}'.format(imgsz_list[idx][0], m[:m.find('-')])
        result_path = osp.join(result_root, exp_id, 'avg_fps.txt')
        avg_fps = np.loadtxt(result_path)
        # avg_fps_dict['{}_{}'.format(imgsz_list[idx][0], m[:m.find('-')])] = avg_fps
        avg_fps_dict['{}'.format(cnt)] = avg_fps
        cnt += 1

print(avg_fps_dict)
avg_fps_dict_sorted = sorted(avg_fps_dict.items(), key=lambda x: x[1], reverse=True)

config_fps_sorted = [int(k) for k, v in avg_fps_dict_sorted]
print(config_fps_sorted)