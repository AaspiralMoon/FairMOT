import os
import os.path as osp
import numpy as np

imgsz_idx = [0, 1, 2, 3, 4]
model_list = ['full-dla_34', 'half-dla_34', 'quarter-dla_34']
imgsz_list = [(1088, 608), (864, 480), (704, 384), (640, 352), (576, 320)]
# for idx in imgsz_idx:
#     for m in model_list:
#         if m == 'full-dla_34':
#             cmd_str = 'python track_half.py --task mot_multiknob --exp_id profiling_{}_{} --load_model ../models/{}.pth --arch {} --imgsize_index {} --is_profiling True'.format(imgsz_list[idx][0], m[:m.find('-')], m, m, idx)
#         else:
#             cmd_str = 'python track_half.py --task mot --exp_id profiling_{}_{} --load_model ../models/{}.pth --arch {} --imgsize_index {} --is_profiling True'.format(imgsz_list[idx][0], m[:m.find('-')], m, m, idx)
#         os.system(cmd_str)

result_root = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/results'
avg_fps_dict = {}
cnt = 0
for idx in imgsz_idx:
    for m in model_list:
        result_path = osp.join(result_root, 'profiling_{}_{}'.format(imgsz_list[idx][0], m[:m.find('-')]), 'avg_fps.txt')
        avg_fps = np.loadtxt(result_path)
        # avg_fps_dict['{}_{}'.format(imgsz_list[idx][0], m[:m.find('-')])] = avg_fps
        avg_fps_dict['{}'.format(cnt)] = avg_fps
        cnt += 1

avg_fps_dict_sorted = sorted(avg_fps_dict.items(), key=lambda x: x[1], reverse=True)

config_fps_sorted = [int(k) for k, v in avg_fps_dict_sorted]
print(config_fps_sorted)