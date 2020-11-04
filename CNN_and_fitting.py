# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 08:25:41 2020

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
from scipy import optimize
from sklearn.decomposition import PCA

#time count
start = time.perf_counter()

_RESIDUAL_THRESHOLD = 30
# _RESIDUAL_THRESHOLD = 2

#Test1nThbg6kXUpJWGl7E1IGOCspRomTxdCARLviKw6E5SY8
# imgfile1 = 'df-ms-data/1/df-googleearth-1k-20091227.jpg'
# imgfile2 = 'df-ms-data/1/df-googleearth-1k-20181029.jpg'
# imgfile1 = 'df-ms-data/1/df-uav-sar-500.jpg'
# imgfile2 = 'df-ms-data/4/3_28.tif'
# imgfile1 = 'df-ms-data/4/3_sar_28.tif'
# imgfile1 = 'df-ms-data/4/3_sar_28.tif'

filename = '1_2'
# path = './datasets/WarmupDataset/' 
path = './datasets/stage1/train/' 
# path = './datasets/stage1/test1/' 

tmp = filename.split('_')
filename_sar =  filename.split('_')[0] + '_sar_' + filename.split('_')[1]

# path_label = path + 'Label/' + filename_sar + '.txt'
imgfile2 = path + 'optical/' + filename + '.tif'
# imgfile2 = path + 'optical_ffdnet_gray/' + filename + '.tif'
imgfile1 = path + 'sar/' + filename_sar + '.tif'
# imgfile1 = path + 'sar_ffdnet_gray/' + filename_sar + '.tif'
# imgfile1 = path + 'sar_loss_L1_view_L1/' + filename_sar + '.tif'

start = time.perf_counter()

# read left image
image1 = imageio.imread(imgfile1)
# 给SAR 图像增加高斯滤波
image1 = cv2.GaussianBlur(image1,(5,5),0)
image2 = imageio.imread(imgfile2)
print(imgfile2, imgfile1)
print('read image time is %6.3f' % (time.perf_counter() - start))

start0 = time.perf_counter()

kps_left, sco_left, des_left = cnn_feature_extract(image1,  nfeatures = -1)
kps_right, sco_right, des_right = cnn_feature_extract(image2,  nfeatures = -1)

print('Feature_extract time is %6.3f, left: %6.3f,right %6.3f' % ((time.perf_counter() - start), len(kps_left), len(kps_right)))
start = time.perf_counter()


##############################################################################
#Flann特征匹配
def calculate_point(kps_left, sco_left, des_left, kps_right, sco_right, des_right):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_left, des_right, k=2)
    
    # matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    # matches = matcher.knnMatch(des_left, des_right, 2)
    
    
    goodMatch = []
    locations_1_to_use = []
    locations_2_to_use = []
    dis = []
    # 匹配对筛选
    min_dist = 1000
    max_dist = 0
    disdif_avg = 0
    # 统计平均距离差
    for m, n in matches:
        disdif_avg += n.distance - m.distance
    disdif_avg = disdif_avg / len(matches)
    # print('disdif_avg:', disdif_avg)
    for m, n in matches:
        #自适应阈值
        if n.distance > m.distance + 1*disdif_avg:
        # if m.distance < 0.9 * n.distance: 
            goodMatch.append(m)
            p2 = cv2.KeyPoint(kps_right[m.trainIdx][0],  kps_right[m.trainIdx][1],  1)
            p1 = cv2.KeyPoint(kps_left[m.queryIdx][0], kps_left[m.queryIdx][1], 1)
            locations_1_to_use.append([p1.pt[0], p1.pt[1]])
            locations_2_to_use.append([p2.pt[0], p2.pt[1]])
            dis.append([n.distance, m.distance])
    #goodMatch = sorted(goodMatch, key=lambda x: x.distance)
    # print('match num is %d' % len(goodMatch))
    locations_1_to_use = np.array(locations_1_to_use)
    locations_2_to_use = np.array(locations_2_to_use)
    dis = np.array(dis)
    
    # Perform geometric verification using RANSAC.
    _, inliers = measure.ransac((locations_1_to_use, locations_2_to_use),
                              transform.AffineTransform,
                              min_samples=3,
                              residual_threshold=_RESIDUAL_THRESHOLD,
                              max_trials=1000)
    
    print('Found %d inliers' % sum(inliers))
    
    inlier_idxs = np.nonzero(inliers)[0]
    
    # 筛选距离最近的前60%个数据
    inlier_idxs = np.nonzero(inliers)[0]
    dis_R = dis[inliers]
    dis_idx = np.argsort(dis_R[:, 0] - dis_R[:, 1])
    dis_sorted = dis_R[dis_idx]
    l = dis_idx.shape[0]
    end = int(l*0.5)
    
    #最终匹配结果
    inlier_idxs = inlier_idxs[dis_idx[:end]]
    
    print('sorted inliers:', end)
    
    #最终匹配结果
    matches = np.column_stack((inlier_idxs, inlier_idxs))
    # print('whole time is %6.3f' % (time.perf_counter() - start0))
    
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
    # print('inlier_idxs:', len(inlier_idxs))
    res = locations_2_to_use[inlier_idxs] - locations_1_to_use[inlier_idxs]
    return locations_1_to_use[inlier_idxs], locations_2_to_use[inlier_idxs]


    

# ############################################################################
# 计算匹配点
res = []
px_cum = None
py_cum = None
cnt = 0
for i in range(200):
    p1, p2 = calculate_point(kps_left, sco_left, des_left, kps_right, sco_right, des_right)
    # plt.scatter(p1[:,0], p2[:,0])
    # plt.scatter(p1[:,1], p2[:,1])
    # points.append(p)
    # print(i, p)

    # np.mean(p2[:,0] - p1[:,0])
    # np.mean(p2[:,1] - p1[:,1])
    
    ##########################################################################
    # 计算x坐标主方向上的斜率
    p = np.hstack((p1, p2))
    px = p[:,[0,2]]
    pca = PCA(n_components=2)
    P_x = pca.fit(px)
    k_x  = P_x.components_[0][1] / P_x.components_[0][0]
    
    
    # 计算y坐标主方向上的斜率
    py = p[:,[1,3]]
    pca = PCA(n_components=2)
    P_y = pca.fit(py)
    # 计算主方向上的斜率
    k_y  = P_y.components_[0][1] / P_y.components_[0][0]
    
    if np.abs(k_x - 1) < 0.07 and np.abs(k_y - 1) < 0.07 and (px[:,1]-px[:,0]).std() < 13 and  (py[:,1]-py[:,0]).std() < 13:
    # if np.abs(k_x - 1) < 0.15 and np.abs(k_y - 1) < 0.15 and (px[:,1]-px[:,0]).std() < 25 and  (py[:,1]-py[:,0]).std() < 25:

        # cnt += 1
        # print('k_x:', k_x)
        # print('k_y:', k_y)
        
        # px_cum.append(px)
        # py_cum.append(py)

        ##########################################################################
        # 拟合数据
        def f_1(x, B):
            return x + B
            
        plt.figure()
        # 拟合点
        x0 = p1[:,0]
        y0 = p2[:,0]
        # 绘制散点
        plt.scatter(x0[:], y0[:], 3, "red")
        # 直线拟合与绘制
        B0 = optimize.curve_fit(f_1, x0, y0)[0]
        x0_hat = np.arange(10,500)
        y0_hat =  x0_hat + B0
        plt.plot(x0_hat, y0_hat, "blue")
        
        plt.figure()
        # 拟合点
        x1 = p1[:,1]
        y1 = p2[:,1]
        # 绘制散点
        plt.scatter(x1[:], y1[:], 3, "red")
        # 直线拟合与绘制
        B1 = optimize.curve_fit(f_1, x1, y1)[0]
        x1_hat = np.arange(10, 500)
        y1_hat =  x1_hat + B1
        plt.plot(x1_hat, y1_hat, "blue")
        
        if 0 < B0 < 288 and 0 < B0 < 288:
            print('B0',B0,'B1:',B1)
            print('x std:', (px[:,1]-px[:,0]).std(), 'y std:', (py[:,1]-py[:,0]).std())
            print('k_x:', k_x, 'k_y:', k_y)
            res.append([B0, B1])
            cnt += 1
            print(cnt)
            
            if cnt == 1:
                px_cum = px
                py_cum = py
            else:
                px_cum = np.vstack((px_cum, px))
                py_cum = np.vstack((py_cum, py))
            
    if cnt >= 30:
        break
print(res)
if len(res) != 0:
    res = np.array(res)
    res = res[:,:,0]
    print(np.mean(res, axis=0))
# points = np.array(points)
# ans = np.mean(points, axis=0)



