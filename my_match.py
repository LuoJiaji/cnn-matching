# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 18:54:29 2020

@author: Bllue
"""

import argparse
import cv2
import numpy as np
import imageio
import plotmatch
from lib.cnn_feature import cnn_feature_extract
import matplotlib.pyplot as plt
import time
from skimage import measure
from skimage import transform

#time count
start = time.perf_counter()

_RESIDUAL_THRESHOLD = 30
#Test1nThbg6kXUpJWGl7E1IGOCspRomTxdCARLviKw6E5SY8
# imgfile1 = 'df-ms-data/1/df-googleearth-1k-20091227.jpg'
# imgfile2 = 'df-ms-data/1/df-googleearth-1k-20181029.jpg'
# imgfile1 = 'df-ms-data/1/df-uav-sar-500.jpg'
# imgfile2 = 'df-ms-data/4/3_28.tif'
# imgfile1 = 'df-ms-data/4/3_sar_28.tif'
# imgfile1 = 'df-ms-data/4/3_sar_28.tif'

filename = '2_47'
# path = './datasets/WarmupDataset/' 
path = './datasets/stage1/train/' 
# path = './datasets/stage1/test1/' 

tmp = filename.split('_')
filename_sar =  filename.split('_')[0] + '_sar_' + filename.split('_')[1]

# path_label = path + 'Label/' + filename_sar + '.txt'
imgfile2 = path + 'optical/' + filename + '.tif'
imgfile1 = path + 'sar/' + filename_sar + '.tif'

start = time.perf_counter()

# read left image
image1 = imageio.imread(imgfile1)
image2 = imageio.imread(imgfile2)
print(imgfile2, imgfile1)
print('read image time is %6.3f' % (time.perf_counter() - start))

start0 = time.perf_counter()

kps_left, sco_left, des_left = cnn_feature_extract(image1,  nfeatures = -1)
kps_right, sco_right, des_right = cnn_feature_extract(image2,  nfeatures = -1)

print('Feature_extract time is %6.3f, left: %6.3f,right %6.3f' % ((time.perf_counter() - start), len(kps_left), len(kps_right)))
start = time.perf_counter()



##############################################################################
des_left = des_left/np.max(des_left) * 255
des_left = des_left.astype('uint8')
des_right = des_right/np.max(des_right) * 255
des_right = des_right.astype('uint8')
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# # KnnMatch()不直接返回匹配结果，而是返回最接近的前K个结果
# # 前K个结果中出现次数最多的结果为最终的匹配结果或者距离最近
matches =  bf.knnMatch(des_left, des_left, k=1)
matches=sorted(matches, key=lambda x:x[0].distance) #据距离来排序

goodMatch = []
locations_1_to_use = []
locations_2_to_use = []
for m in matches:
    #自适应阈值
    # if n.distance > m.distance + 1.7*disdif_avg:
    # if m.distance < 0.97 * n.distance: 
    goodMatch.append(m)
    p2 = cv2.KeyPoint(kps_right[m[0].trainIdx][0],  kps_right[m[0].trainIdx][1],  1)
    p1 = cv2.KeyPoint(kps_left[m[0].queryIdx][0], kps_left[m[0].queryIdx][1], 1)
    locations_1_to_use.append([p1.pt[0], p1.pt[1]])
    locations_2_to_use.append([p2.pt[0], p2.pt[1]])

locations_1_to_use = np.array(locations_1_to_use)
locations_2_to_use = np.array(locations_2_to_use)

# Perform geometric verification using RANSAC.
_, inliers = measure.ransac((locations_1_to_use, locations_2_to_use),
                          transform.AffineTransform,
                          min_samples=3,
                          residual_threshold=_RESIDUAL_THRESHOLD,
                          max_trials=1000)

print('Found %d inliers' % sum(inliers))

inlier_idxs = np.nonzero(inliers)[0]
#最终匹配结果
matches = np.column_stack((inlier_idxs, inlier_idxs))
print('whole time is %6.3f' % (time.perf_counter() - start0))


# Visualize correspondences, and save to file.
#1 绘制匹配连线
plt.rcParams['savefig.dpi'] = 100 #图片像素
plt.rcParams['figure.dpi'] = 100 #分辨率
plt.rcParams['figure.figsize'] = (4.0, 3.0) # 设置figure_size尺寸
_, ax = plt.subplots()
plotmatch.plot_matches(
    ax,
    image1,
    image2,
    locations_1_to_use,
    locations_2_to_use,
    np.column_stack((inlier_idxs, inlier_idxs)),
    plot_matche_points = False,
    matchline = True,
    matchlinewidth = 0.5)
ax.axis('off')
ax.set_title('')
plt.show()