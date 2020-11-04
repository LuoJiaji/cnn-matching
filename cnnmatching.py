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
# _RESIDUAL_THRESHOLD = 2

#Test1nThbg6kXUpJWGl7E1IGOCspRomTxdCARLviKw6E5SY8
# imgfile1 = 'df-ms-data/1/df-googleearth-1k-20091227.jpg'
# imgfile2 = 'df-ms-data/1/df-googleearth-1k-20181029.jpg'
# imgfile1 = 'df-ms-data/1/df-uav-sar-500.jpg'
# imgfile2 = 'df-ms-data/4/3_28.tif'
# imgfile1 = 'df-ms-data/4/3_sar_28.tif'
# imgfile1 = 'df-ms-data/4/3_sar_28.tif'

filename = '1_1'
# path = './datasets/WarmupDataset/' 
# path = './datasets/stage1/train/' 
path = './datasets/stage1/test1/' 

tmp = filename.split('_')
filename_sar =  filename.split('_')[0] + '_sar_' + filename.split('_')[1]

# path_label = path + 'Label/' + filename_sar + '.txt'
imgfile2 = path + 'optical/' + filename + '.tif'
# imgfile1 = path + 'sar/' + filename_sar + '.tif'
# imgfile1 = path + 'sar_ffdnet_gray/' + filename_sar + '.tif'
imgfile1 = path + 'sar_loss_L1_view_L1/' + filename_sar + '.tif'
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
        if n.distance > m.distance + 1.5*disdif_avg:
        # if m.distance < 0.9 * n.distance: 
            goodMatch.append(m)
            p2 = cv2.KeyPoint(kps_right[m.trainIdx][0],  kps_right[m.trainIdx][1],  1)
            p1 = cv2.KeyPoint(kps_left[m.queryIdx][0], kps_left[m.queryIdx][1], 1)
            locations_1_to_use.append([p1.pt[0], p1.pt[1]])
            locations_2_to_use.append([p2.pt[0], p2.pt[1]])
            dis.append([n.distance, m.distance])
    #goodMatch = sorted(goodMatch, key=lambda x: x.distance)
    print('match num is %d' % len(goodMatch))
    locations_1_to_use = np.array(locations_1_to_use)
    locations_2_to_use = np.array(locations_2_to_use)
    dis = np.array(dis)
    # Perform geometric verification using RANSAC.
    _, inliers = measure.ransac((locations_1_to_use, locations_2_to_use),
                              transform.AffineTransform,
                              min_samples=3,
                              residual_threshold=_RESIDUAL_THRESHOLD,
                              max_trials=1000)
    
    # print('Found %d inliers' % sum(inliers))
    
    # 筛选距离最近的前60%个数据
    inlier_idxs = np.nonzero(inliers)[0]
    dis_R = dis[inliers]
    dis_idx = np.argsort(dis_R[:, 0] - dis_R[:, 1])
    dis_sorted = dis_R[dis_idx]
    l = dis_idx.shape[0]
    end = int(l*0.6)
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
    print('inlier_idxs:', len(inlier_idxs))
    res = locations_2_to_use[inlier_idxs] - locations_1_to_use[inlier_idxs]
    return locations_1_to_use[inlier_idxs], locations_2_to_use[inlier_idxs]
    print(np.min(res[:,0]))
    print(np.max(res[:,0]))
    print(np.min(res[:,1]))
    print(np.max(res[:,1]))
    
    
    # mid0 = np.median(res[:,0])
    # m1 = (res[:,0] >= mid0 - 10) & (res[:,0] < mid0 + 10)
    # mid1 = np.median(res[:,1])
    # m2 = (res[:,1] >= mid1 - 10) & (res[:,1] < mid1 + 10)
    # m = m1 & m2
    # res_filter = res[m]
    # point = np.mean(res, axis = 0)
    # # print('point:',point)
    # w = 0.8
    # m1 = (res[:,0] >= 0) & (res[:,0] < 290)
    # m2 = (res[:,1] >= 0) & (res[:,1] < 290)
    # m = m1 & m2
    # res_filter = res[m]
    # std1 = np.std(res_filter[:,0])
    # std2 = np.std(res_filter[:,1])
    # point1 = np.mean(res_filter, axis = 0)
    # # print('point1:', point1)
    # m11 = (res_filter[:,0] >= point1[0] - 5) & (res_filter[:,0] < point1[0] + 5)
    # m22 = (res_filter[:,1] >= point1[1] - 5) & (res_filter[:,1] < point1[1] + 5)
    # m = m11 & m22 
    
    
    # res_filter2 = res_filter[m]
    # if len(res_filter2) != 0:
    #     point2 = np.mean(res_filter2, axis = 0)
    #     # print('point2:', point2)
    #     y = int(round(point2[0]))
    #     x = int(round(point2[1]))
    # elif len(res_filter) != 0:
    #     y = int(round(point1[0]))
    #     x = int(round(point1[1]))
    # else:
    #     y = int(round(point[0]))
    #     x = int(round(point[1]))
    # # print('adjust point', y, x)
    
    # return np.array([y,x])

points = []
for i in range(30):
    p1, p2 = calculate_point(kps_left, sco_left, des_left, kps_right, sco_right, des_right)
    plt.scatter(p1[:,0], p2[:,0])
    plt.scatter(p1[:,1], p2[:,1])
    # points.append(p)
    # print(i, p)
    
points = np.array(points)
ans = np.mean(points, axis=0)


