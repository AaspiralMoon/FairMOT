import subprocess
import os
import os.path as osp

def mkdir_if_missing(d):
    if not osp.exists(d):
        os.makedirs(d)

def get_video_length(path):
    cmd_str = 'ffprobe "%s" -show_entries format=duration -of compact=p=0:nk=1 -v 0'%path
    gettime = subprocess.check_output(cmd_str, shell=True)
    timeT = int(float(gettime.strip()))
    return timeT

def tune_res(path):
    cmd_str1 = 'python track_half.py'
    cmd_str2 = 'python track_half.py'
    cmd_str3 = 'python track_half.py'
    os.system(cmd_str1)
    os.system(cmd_str2)
    os.system(cmd_str3)

def tune_model(path):
    cmd_str1 = 'python track_half.py'
    cmd_str2 = 'python track_half.py'
    cmd_str3 = 'python track_half.py'
    os.system(cmd_str1)
    os.system(cmd_str2)
    os.system(cmd_str3)

def tune_framerate(path):
    cmd_str1 = 'python track_half.py'
    cmd_str2 = 'python track_half.py'
    cmd_str3 = 'python track_half.py'
    os.system(cmd_str1)
    os.system(cmd_str2)
    os.system(cmd_str3)

video_path='/nfs/u40/xur86/videos/test/MOT16-03-results.mp4'
save_path = '/nfs/u40/xur86/videos/test/clips'
interval = 10
video_length = get_video_length(video_path)
start_time = 0
index=1
while start_time < video_length and interval <= video_length:
    mkdir_if_missing(save_path)
    cmd_str = 'ffmpeg -ss %s -i %s -c copy -t %s %s.mp4 -loglevel quiet -y'%(start_time, video_path, interval,'%s/clip%s'%(save_path, index))
    print(cmd_str)
    returnCmd = subprocess.call(cmd_str, shell=True)
    start_time += interval
    index += 1



# def main(video_root):
#     cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(osp.join(result_root, 'frame'), output_video_path)
#     os.system(cmd_str)




# if __name__ == '__main__':
#     video_root = '/nfs/u40/xur86/videos/MOT16-03-results.mp4'
#     main(video_root=video_root,
#          )
