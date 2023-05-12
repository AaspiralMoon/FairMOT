# This script is for tracking on the client and the server
# Author: Renjie Xu
# Time: 2023/5/11

import subprocess
import time

args = '--exp_id test_C2_client_server_arch1 \
        --task mot_multiknob \
        --load_model /nfs/u40/xur86/projects/DeepScale/FairMOT/exp/mot_multiknob/multiknob_res_and_model_full_crowdhuman_multires_freeze_real_1.00_1200/model_1101.pth \
        --load_half_model ../models/half-dla_34.pth \
        --load_quarter_model ../models/quarter-dla_34.pth \
        --switch_period 40 --threshold_config C2'

client_ip = 'wiser-renjie@130.113.70.171'
server_script = 'python track_client_arch1.py {}'.format(args)
client_script = 'CUDA_VISIBLE_DEVICES=3 python track_server_arch1.py {}'.format(args)

# Run server script on the local machine (server)
subprocess.Popen(server_script, shell=True)
time.sleep(3)
# Run client script on the remote machine (client)
subprocess.call(f'ssh {client_ip} {client_script}', shell=True)











# import subprocess
# import time


# args = '--exp_id test_C2_client_server_arch1 \
#         --task mot_multiknob \
#         --load_model /nfs/u40/xur86/projects/DeepScale/FairMOT/exp/mot_multiknob/multiknob_res_and_model_full_crowdhuman_multires_freeze_real_1.00_1200/model_1101.pth \
#         --load_half_model ../models/half-dla_34.pth \
#         --load_quarter_model ../models/quarter-dla_34.pth \
#         --switch_period 40 --threshold_config C2'

# client_str = 'CUDA_VISIBLE_DEVICES=3 python track_client_arch1.py {}'.format(args)
# server_str = 'CUDA_VISIBLE_DEVICES=3 python track_server_arch1.py {}'.format(args)

# # Start the server process
# server_process = subprocess.Popen(server_str, shell=True)

# # Add a small delay to ensure the server starts first
# time.sleep(2)

# # Start the client process
# client_process = subprocess.Popen(client_str, shell=True)

# # Wait for the client process to complete
# client_process.wait()

# # Wait for the server process to complete
# server_process.wait()
