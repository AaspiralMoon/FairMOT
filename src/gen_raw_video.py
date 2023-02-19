# decode jpeg images to raw RGB images, then encode them into a "raw" video
# empty a directory: find /path/to/folder -type f -delete

import cv2
import os
import os.path as osp

def jpeg2raw(input_dir, output_dir):
    # loop through all the JPEG files in the input directory
    for file in os.listdir(input_dir):
        if file.endswith('.jpg') or file.endswith('.jpeg'):
            # load the JPEG image
            img = cv2.imread(os.path.join(input_dir, file), cv2.IMREAD_UNCHANGED)

            # # convert the decompressed image to raw RGB format
            # rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # save the raw RGB image to the output directory
            cv2.imwrite(os.path.join(output_dir, file.replace('.jpg', '.png')), img)
            # with open(osp.join(output_dir, file.replace('.jpg', '.raw')), 'wb') as f:
            #     f.write(rgb_img.tobytes())

def raw2video(input_dir, output_dir):
    cmd_str = 'ffmpeg -i {}/%06d.png -c:v libx264 -preset veryslow -qp 0 {}/output.mp4'.format(input_dir, output_dir)
    os.system(cmd_str)


if __name__ == '__main__':
    seqs = ['MOT17-02-SDP',
            'MOT17-04-SDP',
            'MOT17-05-SDP',
            'MOT17-09-SDP',
            'MOT17-10-SDP',
            'MOT17-11-SDP',
            'MOT17-13-SDP',]

    jpeg_root = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/train/'
    raw_root = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17_multiknob/'
    os.system('find {} -type f -delete'.format(raw_root))

    for seq in seqs:
        input_dir = osp.join(jpeg_root, seq, 'img1')
        output_dir = osp.join(raw_root, seq, 'raw', 'images')
        jpeg2raw(input_dir, output_dir)

    for seq in seqs:
        input_dir = osp.join(raw_root, seq, 'raw', 'images')
        output_dir = osp.join(raw_root, seq, 'raw', 'video')
        raw2video(input_dir, output_dir)