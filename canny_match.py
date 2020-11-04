# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 16:07:13 2020

@author: Bllue
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt 
from skimage import measure
from skimage import transform
import plotmatch

img = cv2.imread("./datasets/stage1/test1/sar/1_sar_1.tif")
# img = cv2.imread("./datasets/stage1/test1/sar_ffdnet_gray/1_sar_1.tif")

img = cv2.GaussianBlur(img,(11,11),0)
img_sar_canny = cv2.Canny(img, 80, 150)
# cv2.imshow('canny', img_canny)
# cv2.waitKey()

img = cv2.imread("./datasets/stage1/test1/optical/1_1.tif")
img = cv2.GaussianBlur(img,(7,7),0)
img_optical_canny = cv2.Canny(img, 30, 60)
# cv2.imshow('canny', img_canny)
# cv2.waitKey()

sift=cv2.xfeatures2d.SIFT_create()#创建sift检测器
kp1, des1 = sift.detectAndCompute(img_sar_canny, None)
kp2, des2 = sift.detectAndCompute(img_optical_canny, None)


#设置Flannde参数
FLANN_INDEX_KDTREE=0
indexParams=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
searchParams= dict(checks=50)
flann=cv2.FlannBasedMatcher(indexParams, searchParams)
matches = flann.knnMatch(des1,des2,k=2)
#设置好初始匹配值
matchesMask=[[0,0] for i in range (len(matches))]
locations_1_to_use = []
locations_2_to_use = []
for i, (m,n) in enumerate(matches):
    if m.distance< 0.95*n.distance:  #舍弃小于0.5的匹配结果
        matchesMask[i]=[1,0]
        # print(i)
        # p2 = cv2.KeyPoint(kp1[m.trainIdx][0],  kp1[m.trainIdx][1],  1)
        # p1 = cv2.KeyPoint(kp2[m.queryIdx][0], kp2[m.queryIdx][1], 1)
        locations_1_to_use.append([kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1]])
        locations_2_to_use.append([kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]])
locations_1_to_use = np.array(locations_1_to_use)
locations_2_to_use = np.array(locations_2_to_use)
# Perform geometric verification using RANSAC.
_RESIDUAL_THRESHOLD = 30

_, inliers = measure.ransac((locations_1_to_use, locations_2_to_use),
                          transform.AffineTransform,
                          min_samples=3,
                          residual_threshold=_RESIDUAL_THRESHOLD,
                          max_trials=1000)
inlier_idxs = np.nonzero(inliers)[0]
# p1 = locations_2_to_use[inlier_idxs] 
# p2 = locations_1_to_use[inlier_idxs]


# Visualize correspondences, and save to file.
#1 绘制匹配连线
plt.rcParams['savefig.dpi'] = 100 #图片像素
plt.rcParams['figure.dpi'] = 100 #分辨率
plt.rcParams['figure.figsize'] = (4.0, 3.0) # 设置figure_size尺寸
_, ax = plt.subplots()
plotmatch.plot_matches(
    ax,
    img_sar_canny,
    img_optical_canny,
    locations_1_to_use,
    locations_2_to_use,
    np.column_stack((inlier_idxs, inlier_idxs)),
    plot_matche_points = False,
    matchline = True,
    matchlinewidth = 0.5)
ax.axis('off')
ax.set_title('')
plt.show()