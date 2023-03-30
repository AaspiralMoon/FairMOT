# this script is for testing any code
# Author: Renjie Xu
# Time: 2023/2/22

# import _init_paths
# import torch
# import os
# import os.path as osp
# import numpy as np
# import numpy as np
# import matplotlib
# import time
# import matplotlib.pyplot as plt
# import cv2

# detection_result_path = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17_multiknob/results'
# save_path = '/nfs/u40/xur86/projects/DeepScale/FairMOT/exp/mot_multiknob/verify_labels/verify_detections'

# def plot_label(img, labels):
#     img = cv2.imread(img)
#     matplotlib.use('Agg')
#     plt.close('all')
#     plt.figure()
#     plt.imshow(img[:, :, ::-1])
#     plt.plot(labels[:, [0, 2, 2, 0, 0]].T, labels[:, [1, 1, 3, 3, 1]].T, '.-')
#     plt.axis('off')
#     plt.savefig(osp.join(save_path, 'test.jpg'))
#     time.sleep(3)
#     plt.close('all')

# img = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17_multiknob/train/MOT17-13-SDP/img1/000001.jpg'
# labels = np.loadtxt('/nfs/u40/xur86/projects/DeepScale/datasets/MOT17_multiknob/results/MOT17-13-SDP/576_quarter/1.txt')
# plot_label(img, labels)

# avg_time = [0.5]
# avg_time_array = np.asarray(avg_time)
# result_root = '/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/results'
# path = osp.join(result_root, 'avg_fps.txt')
# print(avg_time_array)
# print(path)
# np.savetxt(path, 1.0 / avg_time_array, fmt='%.2f')

# a = [11, 14, 8, 13, 10, 7, 5, 4, 12, 9, 2, 6, 1, 3, 0]
# b = [2, 4, 5, 6, 10, 11, 12]

# result = min((a.index(number), number) for number in b)[1]
# aaa = [(a.index(number), number) for number in b]
# print(min(aaa))
# import numpy as np
# import matplotlib.pyplot as plt

# # Generate Pareto front points
# pareto_points = np.array([
#     [5, 25],
#     [15, 45],
#     [30, 65],
#     [45, 80],
#     [60, 90],
#     [75, 95],
#     [90, 98],
# ])

# # Generate random points below the Pareto front
# num_random_points = 50
# random_points = np.zeros((num_random_points, 2))

# for i in range(num_random_points):
#     x = np.random.uniform(pareto_points[0, 0], pareto_points[-1, 0])
#     y = np.random.uniform(0, np.interp(x, pareto_points[:, 0], pareto_points[:, 1]) - 1)
#     random_points[i] = [x, y]

# # Combine the Pareto front points and random points
# points = np.concatenate((pareto_points, random_points), axis=0)

# # Plot the points and Pareto front
# plt.scatter(random_points[:, 0], random_points[:, 1], label="Below the curve")
# plt.plot(pareto_points[:, 0], pareto_points[:, 1], color="red", marker='o', label="Pareto front")
# plt.xlabel("Latency (ms)")
# plt.ylabel("Accuracy (%)")
# plt.xlim(0, 100)
# plt.ylim(0, 100)
# plt.legend()
# plt.title("Pareto Front")

# # Save the plot as an image
# plt.savefig('pareto_front.png', dpi=300)

# import numpy as np
# import matplotlib.pyplot as plt

# # Generate Pareto front points
# pareto_points = np.array([
#     [5, 25],
#     [15, 45],
#     [30, 65],
#     [45, 80],
#     [60, 90],
#     [75, 95],
#     [90, 98],
# ])

# # Generate random points below the Pareto front
# num_random_points = 50
# random_points = np.zeros((num_random_points, 2))

# for i in range(num_random_points):
#     x = np.random.uniform(pareto_points[0, 0], pareto_points[-1, 0])
#     y = np.random.uniform(0, np.interp(x, pareto_points[:, 0], pareto_points[:, 1]) - 1)
#     random_points[i] = [x, y]

# # Combine the Pareto front points and random points
# points = np.concatenate((pareto_points, random_points), axis=0)

# # Plot the points and Pareto front
# plt.scatter(random_points[:, 0], random_points[:, 1])
# plt.plot(pareto_points[:, 0], pareto_points[:, 1], color="red", marker='o', label="Pareto frontier")

# # Label the points on the curve
# point_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
# for i, txt in enumerate(point_labels):
#     plt.annotate(txt, (pareto_points[i, 0] - 2, pareto_points[i, 1] + 1))

# plt.xlabel("Latency (ms)")
# plt.ylabel("Accuracy (%)")
# plt.xlim(0, 100)
# plt.ylim(0, 100)
# plt.legend()
# plt.title("Pareto Frontier")

# # Save the plot as an image
# plt.savefig('pareto_front.png', dpi=300)

thresholds_preset = [0.61, 0.66, 0.71, 0.62, 0.67, 0.72, 0.63, 0.68, 0.73, 0.64, 0.69, 0.74, 0.65, 0.70, 0.75]

thresholds = [x + 5*0.05 for x in thresholds_preset]
print(thresholds)