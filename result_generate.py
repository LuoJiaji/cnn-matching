import os
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

def calculate_point_MCMC(kps_left, sco_left, des_left, kps_right, sco_right, des_right):
    #Flann特征匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=40)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_left, des_right, k=2)

    goodMatch = []
    locations_1_to_use = []
    locations_2_to_use = []
    dis = []
    # 匹配对筛选
    # min_dist = 1000
    # max_dist = 0
    disdif_avg = 0
    # 统计平均距离差
    for m, n in matches:
        disdif_avg += n.distance - m.distance
    disdif_avg = disdif_avg / len(matches)

    for m, n in matches:
        #自适应阈值
        if n.distance > m.distance + 1*disdif_avg:
            goodMatch.append(m)
            p2 = cv2.KeyPoint(kps_right[m.trainIdx][0],  kps_right[m.trainIdx][1],  1)
            p1 = cv2.KeyPoint(kps_left[m.queryIdx][0], kps_left[m.queryIdx][1], 1)
            locations_1_to_use.append([p1.pt[0], p1.pt[1]])
            locations_2_to_use.append([p2.pt[0], p2.pt[1]])
            dis.append([n.distance, m.distance])
    # if
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

    # print('Found %d inliers' % sum(inliers))

    inlier_idxs = np.nonzero(inliers)[0]

    # 筛选距离最近的前60%个数据
    inlier_idxs = np.nonzero(inliers)[0]
    dis_R = dis[inliers]
    dis_idx = np.argsort(dis_R[:, 0] - dis_R[:, 1])
    dis_sorted = dis_R[dis_idx]
    l = dis_idx.shape[0]
    end = int(l*0.6)
    
    #最终匹配结果
    inlier_idxs = inlier_idxs[dis_idx[:end]]
    
    # print('sorted inliers:', end)

    #最终匹配结果
    matches = np.column_stack((inlier_idxs, inlier_idxs))
    # print('whole time is %6.3f' % (time.perf_counter() - start0))
    
    return locations_1_to_use[inlier_idxs], locations_2_to_use[inlier_idxs]
    # Visualize correspondences, and save to file.
    #1 绘制匹配连线
    # plt.rcParams['savefig.dpi'] = 100 #图片像素
    # plt.rcParams['figure.dpi'] = 100 #分辨率
    # plt.rcParams['figure.figsize'] = (4.0, 3.0) # 设置figure_size尺寸
    # _, ax = plt.subplots()
    # plotmatch.plot_matches(
    #     ax,
    #     image1,
    #     image2,
    #     locations_1_to_use,
    #     locations_2_to_use,
    #     np.column_stack((inlier_idxs, inlier_idxs)),
    #     plot_matche_points = False,
    #     matchline = True,
    #     matchlinewidth = 0.5)
    # ax.axis('off')
    # ax.set_title('')
    # plt.show()

    # res = locations_2_to_use[inlier_idxs] - locations_1_to_use[inlier_idxs]
    # point = np.mean(res, axis = 0)
    # # print('point:', point)
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
    #     # print('point2', point2)
    #     y = int(round(point2[0]))
    #     x = int(round(point2[1]))
    # elif len(res_filter) != 0:
    #     y = int(round(point1[0]))
    #     x = int(round(point1[1]))
    # else:
    #     y = int(round(point[0]))
    #     x = int(round(point[1]))
    # # print('adjust point', y, x)
    # print(y,x)
    # return np.array([y,x])



path = './datasets/stage1/test1/' 
# path = './datasets/stage1/train/' 
path_list_optical = os.listdir(path + '/optical')

path_list_sar = []

for path_optical in path_list_optical:
    path_sar = path_optical.split('_')[0] + '_sar_' + path_optical.split('_')[1]
    path_list_sar.append(path_sar)

print('number of data:', len(path_list_sar))

t = time.strftime("%Y%m%d%H%M%S", time.localtime()) 
res_file = './result/result'+ t + '.txt'
f = open(res_file,'w')
f.write(str(len(path_list_optical))+ '\n')
f.flush()

yichang_file = './result/yichang'+ t + '.txt'
f_yichang = open(yichang_file,'w')
# f_yichang.write(str(len(path_list_optical))+ '\n')
# f.flush()
log_file = './result/log'+ t + '.txt'
f_log = open(log_file,'w')

for  i in range(300):
    #time count
    start = time.perf_counter()

    _RESIDUAL_THRESHOLD = 30
    #Test1nThbg6kXUpJWGl7E1IGOCspRomTxdCARLviKw6E5SY8
    # imgfile1 = 'df-ms-data/1/df-googleearth-1k-20091227.jpg'
    # imgfile2 = 'df-ms-data/1/df-googleearth-1k-20181029.jpg'
    # imgfile1 = 'df-ms-data/1/df-uav-sar-500.jpg'
    
    # imgfile2 = 'df-ms-data/4/3_28.tif'
    # imgfile1 = 'df-ms-data/4/3_sar_28.tif'
    
    imgfile2 = path + 'optical/' + path_list_optical[i]
    # imgfile1 = path + 'sar/' + path_list_sar[i]
    imgfile1 = path + 'sar_loss_L1_view_L1/' + path_list_sar[i]

    start = time.perf_counter()

    # read left image
    image1 = imageio.imread(imgfile1)
    image2 = imageio.imread(imgfile2)
    print('*' * 40)
    print(imgfile2, imgfile1)
    f_log.write(imgfile1 + ' ' + imgfile2 + '\n')
    f_log.flush()
    print('read image time is %6.3f' % (time.perf_counter() - start))

    start0 = time.perf_counter()

    kps_left, sco_left, des_left = cnn_feature_extract(image1,  nfeatures = -1)
    kps_right, sco_right, des_right = cnn_feature_extract(image2,  nfeatures = -1)

    print('Feature_extract time is %6.3f, left: %6.3f,right %6.3f' % ((time.perf_counter() - start), len(kps_left), len(kps_right)))
    start = time.perf_counter()
    
    # ############################################################################
    # 计算匹配点
    res = []
    px_cum = None
    py_cum = None
    cnt = 0
    for _ in range(500):
        p1, p2 = calculate_point_MCMC(kps_left, sco_left, des_left, kps_right, sco_right, des_right)
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
        # 判断x,y的斜率在0.9到1.1之间,坐标差值的标准差小于30个像素，则认为匹配准确，并进行拟合定位
        if np.abs(k_x - 1) < 0.05 and np.abs(k_y - 1) < 0.05 and (px[:,1]-px[:,0]).std() < 15 and  (py[:,1]-py[:,0]).std() < 15:
            # cnt += 1
            # print('k_x:', k_x)
            # print('k_y:', k_y)
            
            # px_cum.append(px)
            # py_cum.append(py)

            ##########################################################################
            # 拟合数据
            def f_1(x, B):
                return x + B
                
            # plt.figure()
            # 拟合点
            x0 = p1[:,0]
            y0 = p2[:,0]
            # 绘制散点
            # plt.scatter(x0[:], y0[:], 3, "red")
            # 直线拟合与绘制
            B0 = optimize.curve_fit(f_1, x0, y0)[0]
            # x0_hat = np.arange(10,500)
            # y0_hat =  x0_hat + B0
            # plt.plot(x0_hat, y0_hat, "blue")
            
            # plt.figure()
            # 拟合点
            x1 = p1[:,1]
            y1 = p2[:,1]
            # 绘制散点
            # plt.scatter(x1[:], y1[:], 3, "red")
            # 直线拟合与绘制
            B1 = optimize.curve_fit(f_1, x1, y1)[0]
            # x1_hat = np.arange(10, 500)
            # y1_hat =  x1_hat + B1
            # plt.plot(x1_hat, y1_hat, "blue")
            if 0 < B0 < 288 and 0 < B0 < 288:
                print('B0',B0,'B1:',B1)
                # print('x std:', (px[:,1]-px[:,0]).std(), 'y std:', (py[:,1]-py[:,0]).std())
                f_log.write('B0:' + str(B0) + '  ' + 'B1:' + str(B1) + '\n')
                # f_log.write('x std:' + str((px[:,1]-px[:,0]).std()) + '  ' +  'y std:' + str((py[:,1]-py[:,0]).std()) + '\n')
                # f_log.write('k_x:' + str(k_x) + '  ' + 'k_y:' + str(k_y) + '\n')
                f_log.flush()
                res.append([B0, B1])
                cnt += 1
                
                if cnt == 1:
                    px_cum = px
                    py_cum = py
                else:
                    px_cum = np.vstack((px_cum, px))
                    py_cum = np.vstack((py_cum, py))

        if cnt >= 50:
            break
    # print(res)
    if len(res) != 0:
        res = np.array(res)
        res = res[:,:,0]
        res = np.mean(res, axis=0)
        # print(res)
        y = int(round(res[0]))
        x = int(round(res[1]))
        print('result:', y, x)
        f.write(path_list_optical[i][0] + ' ' + path_list_optical[i] + ' ' + path_list_sar[i] + ' '+ str(y) + ' '+ str(x) + '\n')
        f.flush()
    else:
        f_yichang.write(path_list_optical[i] + '\n')
        f_yichang.flush()
f.close()
f_yichang.close()
