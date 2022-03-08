from cv2 import mean
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import time

def cal_hist(img, size=(112, 112), channel=2):  # b=0, g=1, r=2
    img = cv2.resize(img, size)
    hist = cv2.calcHist([img], [channel], None, [256], [0, 256])
    return hist

def edge_detection_sobel(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    Scale_absX = cv2.convertScaleAbs(x)  
    Scale_absY = cv2.convertScaleAbs(y)
    result = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
    return result

def edge_detection_scharr(img):
    x = cv2.Scharr(img, cv2.CV_16S, 1, 0) 
    y = cv2.Scharr(img, cv2.CV_16S, 0, 1)
    Scale_absX = cv2.convertScaleAbs(x)
    Scale_absY = cv2.convertScaleAbs(y)
    Scharr = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
    return Scharr

img_dir1 = '/nfs/u40/xur86/datasets/MOT17/images/train/MOT17-02-SDP/img1/000568.jpg'
img_dir2 = '/nfs/u40/xur86/datasets/MOT17/images/train/MOT17-02-SDP/img1/000569.jpg'
img1 = cv2.imread(img_dir1)
img2 = cv2.imread(img_dir2)
edge1 = edge_detection_sobel(img1)
edge2 = edge_detection_sobel(img2)
hist1 = cal_hist(img1)
hist2 = cal_hist(img2)