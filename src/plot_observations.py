import numpy as np
import matplotlib.pyplot as plt


# data1 = np.loadtxt('/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/results/576_half-dla_34_mot17_half_hm/mAP.txt')
# data2 = np.loadtxt('/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/results/640_half-dla_34_mot17_half_hm/mAP.txt')
# data3 = np.loadtxt('/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/results/704_half-dla_34_mot17_half_hm/mAP.txt')
# data4 = np.loadtxt('/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/results/864_half-dla_34_mot17_half_hm/mAP.txt')
# plt.plot(data1[0:200], label = '576*320', color ='b')
# plt.plot(data2[0:200], label = '640*352', color ='g')
# plt.plot(data2[0:200], label = '704*384', color ='y')
# plt.plot(data4[0:200], label = '864*480', color ='r')
# plt.xlabel("Frame")
# plt.ylabel("mAP")
# plt.show()
# plt.legend(loc = 0, ncol = 2)
# plt.savefig('mAP.jpg')


data = np.loadtxt('/nfs/u40/xur86/projects/DeepScale/datasets/MOT17/images/results/half-dla_34_mot17_half_dets.txt')
data1 = data[0: 200, 2]
data2 = data[2284: 2484, 2]
data3 = data[4568: 4768, 2]
data4 = data[6852: 7052, 2]
plt.plot(data1, label = '576*320', color ='b')
plt.plot(data2, label = '640*352', color ='g')
plt.plot(data2, label = '704*384', color ='y')
plt.plot(data4, label = '864*480', color ='r')
plt.xlabel("Frame")
plt.ylabel("hm Per")
plt.show()
plt.legend(loc = 0, ncol = 2)
plt.savefig('hm_Per.jpg')