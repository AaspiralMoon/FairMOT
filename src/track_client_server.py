import subprocess
import time

args = '--task mot --exp_id test_client_server --load_model ../models/full-dla_34.pth'

client_str = 'python track_client.py {}'.format(args)
server_str = 'python track_server.py {}'.format(args)

# Start the server process
server_process = subprocess.Popen(server_str, shell=True)

# Add a small delay to ensure the server starts first
time.sleep(2)

# Start the client process
client_process = subprocess.Popen(client_str, shell=True)

# Wait for the client process to complete
client_process.wait()

# Wait for the server process to complete
server_process.wait()
