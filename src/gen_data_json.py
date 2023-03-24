# generate mot17_multiknob.train

import os
import os.path as osp
import numpy as np

result_path = '/nfs/u40/xur86/projects/DeepScale/FairMOT/src/data/mot17_multiknob.half'
input_path = '/nfs/u40/xur86/projects/DeepScale/FairMOT/src/data/mot17.half'


with open(input_path, 'r') as file:
    for line in file:
        line = line.strip()
        line_new = line.replace('MOT17/images/train/', 'MOT17_multiknob/train/')
        with open(result_path, 'a+') as f:
            f.write(line_new + '\n')