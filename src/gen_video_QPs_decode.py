# Saving the jpg images as png, and encode them into a video, then encode the video at different QPs, finally decode the videos into jpg images.
# Author: Renjie Xu
# Time: 2023/3/23

# empty a directory: find /path/to/folder -type f -delete

import cv2
import os
import os.path as osp

def mkdir_if_missing(d):
    if not osp.exists(d):
        os.makedirs(d)

def jpeg2raw(input_dir, output_dir):
    mkdir_if_missing(output_dir)
    # loop through all the JPEG files in the input directory
    for file in os.listdir(input_dir):
        if file.endswith('.jpg') or file.endswith('.jpeg'):
            img = cv2.imread(os.path.join(input_dir, file), cv2.IMREAD_UNCHANGED)
            cv2.imwrite(os.path.join(output_dir, file.replace('.jpg', '.png')), img)

def raw2video(input_dir, output_dir):
    mkdir_if_missing(output_dir)
    cmd_str = 'ffmpeg -i {}/%06d.png -c:v libx264 -preset veryslow -qp 0 {}/output.mp4'.format(input_dir, output_dir)
    os.system(cmd_str)

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


if __name__ == '__main__':
    seqs = ['MOT17-02-SDP',
            'MOT17-04-SDP',
            'MOT17-05-SDP',
            'MOT17-09-SDP',
            'MOT17-10-SDP',
            'MOT17-11-SDP',
            'MOT17-13-SDP']

    jpeg_root = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/train'
    raw_root = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17_multiknob'
    # qp_list = [25, 30, 35, 40]
    qp_list = [20]
    # for seq in seqs:
    #     print('Cleaning {}'.format(seq))
    #     os.system('rm -rf {}/{}/*'.format(raw_root, seq))

    # for seq in seqs:
    #     input_dir = osp.join(jpeg_root, seq, 'img1')
    #     output_dir = osp.join(raw_root, seq, 'raw', 'images')
    #     jpeg2raw(input_dir, output_dir)

    # for seq in seqs:
    #     input_dir = osp.join(raw_root, seq, 'raw', 'images')
    #     output_dir = osp.join(raw_root, seq, 'raw', 'video')
    #     raw2video(input_dir, output_dir)

    change_video_qp(raw_root, seqs, qp_list)