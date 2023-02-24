# knobs: resolution, quantization parameter, model
# resolution: resize image
# quantization parameter: change QP of videos
# model: -
# extract high quality images from a video: ffmpeg -i MOT17-02-SDP.mp4 -r 30 -q:v 1 -qmin 1 -qmax 1 images/%06d.jpg
# encode images into a video without quality loss: ffmpeg -f image2 -r 30 -i %06d.jpg -vcodec libx264 -profile:v high444 -refs 16 -crf 0 -preset ultrafast MOT17-02-SDP.mp4

import os
import os.path as osp

def mkdir_if_missing(d):
    if not osp.exists(d):
        os.makedirs(d)

def change_video_qp(data_root, seqs, qp_list):
    for seq in seqs:
        for qp in qp_list:
            raw_video_path = osp.join(data_root, seq, 'raw', 'video')
            knob_path = osp.join(data_root, seq, 'QP_{}'.format(qp))
            knob_video_path = osp.join(knob_path, 'video')
            knob_image_path = osp.join(knob_path, 'images')
            mkdir_if_missing(knob_path)
            mkdir_if_missing(knob_video_path)
            mkdir_if_missing(knob_image_path)
            cmd_str = 'ffmpeg -i {}/output.mp4 -c:v libx264 -preset veryslow -qp {} {}/output.mp4'.format(raw_video_path, qp, knob_video_path)
            os.system(cmd_str)
            cmd_str2 = 'ffmpeg -i {}/output.mp4 -q:v 0 {}/%06d.jpg'.format(knob_video_path, knob_image_path)
            os.system(cmd_str2)

def gen_detections(data_root, result_root, model_root, seqs, imgsize_index, model_list, qp_list):
    imgsize_list = [(1088, 608), (864, 480), (704, 384), (640, 352), (576, 320)]
    for seq in seqs:
        for idx in imgsize_index:
            for m in model_list:
                for qp in qp_list:
                    img_path = osp.join(data_root, seq, 'QP_{}'.format(qp), 'images')
                    result_path = osp.join(result_root, seq, '{}_{}_{}'.format(imgsize_list[idx][0], m[:m.find('-')], qp))
                    mkdir_if_missing(result_path)
                    cmd_str = 'python gen_detections.py --data_dir {} --output_root {} --load_model {}/{}.pth --imgsize_index {} --arch {} --gen_dets'.format(img_path, 
                                                                                                                                                            result_path, model_root, m, idx, m)
                    os.system(cmd_str)

if __name__ == '__main__':
    seqs = ['MOT17-02-SDP',
            'MOT17-04-SDP',
            'MOT17-05-SDP',
            'MOT17-09-SDP',
            'MOT17-10-SDP',
            'MOT17-11-SDP',
            'MOT17-13-SDP']
    model_root = '/nfs/u40/xur86/projects/DeepScale/FairMOT/exp/mot/mot17_multiknob'
    data_root = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17_multiknob'
    result_root = osp.join(data_root, 'results')
    mkdir_if_missing(result_root)
    imgsize_index = [0, 1, 2, 3, 4]   # (1088, 608), (864, 480), (704, 384), (640, 352), (576, 320)
    model_list = ['full-dla_34', 'half-dla_34', 'quarter-dla_34']
    qp_list = [10, 20, 30, 40, 50]
    # change_video_qp(data_root, seqs, qp_list)
    gen_detections(data_root, result_root, model_root, seqs, imgsize_index, model_list, qp_list)