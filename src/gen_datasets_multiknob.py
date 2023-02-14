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
            raw_video_path = osp.join(data_root, seq, 'raw', 'video', '{}.mp4'.format(seq))
            knob_path = osp.join(data_root, seq, 'QP_{}'.format(qp))
            knob_video_path = osp.join(knob_path, 'video')
            knob_image_path = osp.join(knob_path, 'images')
            mkdir_if_missing(knob_path)
            mkdir_if_missing(knob_video_path)
            mkdir_if_missing(knob_image_path)
            cmd_str = 'ffmpeg -i {} -c:v libx264 -qp {} {}/{}.mp4'.format(raw_video_path, qp, knob_video_path, seq)
            os.system(cmd_str)
            cmd_str2 = 'ffmpeg -i {}/{}.mp4 -q:v 1 -qmin 1 -qmax 1 {}/%06d.jpg'.format(knob_video_path, seq, knob_image_path)
            os.system(cmd_str2)


if __name__ == '__main__':
    # video_list = ['MOT17-02-SDP.mp4',
    #               'MOT17-04-SDP.mp4',
    #               'MOT17-05-SDP.mp4',
    #               'MOT17-09-SDP.mp4',
    #               'MOT17-10-SDP.mp4',
    #               'MOT17-11-SDP.mp4',
    #               'MOT17-13-SDP.mp4',]
    # seqs = ['MOT17-02-SDP',
    #         'MOT17-04-SDP',
    #         'MOT17-05-SDP',
    #         'MOT17-09-SDP',
    #         'MOT17-10-SDP',
    #         'MOT17-11-SDP',
    #         'MOT17-13-SDP',]
    seqs = ['MOT17-05-SDP']
    data_root = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17_multiknob/'

    resolution_list = [(1088, 608), (864, 480), (704, 384), (640, 352), (576, 320)]
    qp_list = [0, 10, 20, 30, 40, 50]
    model_list = ['full_dla-34, half_dla-34, quarter_dla-34']
    # framerate_list = []
    change_video_res_qp(data_root, seqs, qp_list)