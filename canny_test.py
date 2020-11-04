# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 16:07:13 2020

@author: Bllue
"""
import cv2
import numpy as np
 
def CannyThreshold(lowThreshold):
    detected_edges = cv2.GaussianBlur(gray,(5,5),0)
    detected_edges = cv2.Canny(detected_edges,
                               lowThreshold,
                               lowThreshold*ratio,
                               apertureSize = kernel_size)
    dst = cv2.bitwise_and(img,img,mask = detected_edges)  # just add some colours to edges from original image.
    cv2.imshow('canny demo',dst)

lowThreshold = 0
max_lowThreshold = 200
ratio = 3
kernel_size = 3
 
# img = cv2.imread('qingwen.png')
# img = cv2.imread("./datasets/stage1/test1/sar_ffdnet_gray/1_sar_1.tif")
img = cv2.imread("./datasets/stage1/test1/sar/1_sar_1.tif")

# img = cv2.imread("./img/1_sar_27.tif")
img = cv2.GaussianBlur(img,(3,3),0)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = img/np.max(img) * 255
img = img.astype('uint8')
cv2.namedWindow('canny demo')
 
cv2.createTrackbar('Min threshold','canny demo',lowThreshold, max_lowThreshold, CannyThreshold)
 
CannyThreshold(0)  # initialization
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()