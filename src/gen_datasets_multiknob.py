# knobs: resolution, quantization parameter, model
# resolution: resize image
# quantization parameter: change QP of videos
# model: -

import os
import os.path as osp

def mkdir_if_missing(d):
    if not osp.exists(d):
        os.makedirs(d)

# def change_video_qp(data_root, seqs, qp_list):
#     for seq in seqs:
#         for qp in qp_list:
#             raw_video_path = osp.join(data_root, seq, 'raw', 'video')
#             knob_path = osp.join(data_root, seq, 'QP_{}'.format(qp))
#             knob_video_path = osp.join(knob_path, 'video')
#             knob_image_path = osp.join(knob_path, 'images')
#             mkdir_if_missing(knob_path)
#             mkdir_if_missing(knob_video_path)
#             mkdir_if_missing(knob_image_path)
#             cmd_str = 'ffmpeg -i {}/output.mp4 -c:v libx264 -preset veryslow -qp {} {}/output.mp4'.format(raw_video_path, qp, knob_video_path)
#             os.system(cmd_str)
#             cmd_str2 = 'ffmpeg -i {}/output.mp4 -q:v 0 {}/%06d.jpg'.format(knob_video_path, knob_image_path)
#             os.system(cmd_str2)

# def gen_detections(data_root, result_root, model_root, seqs, imgsize_index, model_list, qp_list):       # for 3 knobs
#     imgsize_list = [(1088, 608), (864, 480), (704, 384), (640, 352), (576, 320)]
#     for seq in seqs:
#         for idx in imgsize_index:
#             for m in model_list:
#                 for qp in qp_list:
#                     img_path = osp.join(data_root, seq, 'QP_{}'.format(qp), 'images')
#                     result_path = osp.join(result_root, seq, '{}_{}_{}'.format(imgsize_list[idx][0], m[:m.find('-')], qp))
#                     mkdir_if_missing(result_path)
#                     cmd_str = 'python gen_detections.py --data_dir {} --output_root {} --load_model {}/{}.pth --imgsize_index {} --arch {} --gen_dets'.format(img_path, 
#                                                                                                                                                             result_path, model_root, m, idx, m)
#                     os.system(cmd_str)

# def gen_detections(data_root, result_root, model_root, seqs, imgsize_index):       # only for imgsize
#     imgsize_list = [(1088, 608), (864, 480), (704, 384), (640, 352), (576, 320)]
#     for seq in seqs:
#         for idx in imgsize_index:
#             img_path = osp.join(data_root, seq, 'QP_0', 'images')
#             result_path = osp.join(result_root, seq, '{}'.format(imgsize_list[idx][0]))
#             mkdir_if_missing(result_path)
#             cmd_str = 'python gen_detections.py --data_dir {} --output_root {} --load_model {} --imgsize_index {} --arch full-dla_34 --gen_dets'.format(img_path, result_path, model_root, idx)
#             os.system(cmd_str)

def gen_detections(data_root, result_root, model_root, seqs, imgsize_index, model_list):       # for imgsize and model
    imgsize_list = [(1088, 608), (864, 480), (704, 384), (640, 352), (576, 320)]
    for seq in seqs:
        for idx in imgsize_index:
            for m in model_list:
                img_path = osp.join(data_root, 'train', seq, 'img1')
                result_path = osp.join(result_root, seq, '{}_{}'.format(imgsize_list[idx][0], m[:m.find('-')]))
                mkdir_if_missing(result_path)
                cmd_str = 'python gen_detections.py --data_dir {} --output_root {} --load_model {}/{}.pth --imgsize_index {} --arch {} --gen_dets'.format(img_path, result_path, model_root, m, idx, m)
                os.system(cmd_str)

if __name__ == '__main__':
    seqs = ['MOT17-02-SDP',
            'MOT17-04-SDP',
            'MOT17-05-SDP',
            'MOT17-09-SDP',
            'MOT17-10-SDP',
            'MOT17-11-SDP',
            'MOT17-13-SDP']
    model_root = '../models'
    data_root = '../../datasets/MOT17_multiknob'
    result_root = osp.join(data_root, 'results_yolo')
    mkdir_if_missing(result_root)
    imgsize_index = [0, 1, 2, 3, 4]   # (1088, 608), (864, 480), (704, 384), (640, 352), (576, 320)
    # model_list = ['full-dla_34', 'half-dla_34', 'quarter-dla_34']
    model_list = ['full-yolo', 'half-yolo', 'quarter-yolo']
    print('Cleaning...')
    cmd_str = 'rm -rf {}/*'.format(result_root)             # delete previous results
    os.system(cmd_str)
    print('Clean up finished!')
    gen_detections(data_root, result_root, model_root, seqs, imgsize_index, model_list)