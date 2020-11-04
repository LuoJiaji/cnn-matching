
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

filename = '1_7'
y, x = 59, 161

# path = './datasets/WarmupDataset/' 
# path = './datasets/stage1/train/' 
path = './datasets/stage1/test1/' 
tmp = filename.split('_')
filename_sar =  filename.split('_')[0] + '_sar_' + filename.split('_')[1]

path_label = path + 'Label/' + filename_sar + '.txt'
path_optical = path + 'optical/' + filename + '.tif'
# path_sar = path + 'sar/' + filename_sar + '.tif'
path_sar = path + 'sar_ffdnet_gray/' + filename_sar + '.tif'
# print(path_label)
# f = open(path_label,"r")   #设置文件对象
# info = f.read()     #将txt文件的所有内容读入到字符串str中
# info = info.split(' ')
# y,x = int(info[2]), int(info[3])
# f.close() 
# print(info)

# plt.ion()
img_optical=Image.open(path_optical)
# print('%s, %s, %s' % (img.mode, img.size, img.format))
img_optical = np.array(img_optical)
print('optical img shape:', img_optical.shape)
# plt.subplot(2,2,1)
plt.figure()
plt.imshow(img_optical, cmap='gray')
# plt.imshow(img)
plt.title('original optical img')
# plt.show()

plt.figure()
img_optical_crop = img_optical[x:x+512, y:y+512]
print('croped optical img shape:', img_optical_crop.shape)
plt.subplot(1,2,1)
plt.imshow(img_optical_crop, cmap='gray')
plt.title('croped optical img')
# plt.show()


img_sar = Image.open(path_sar)
img_sar = np.array(img_sar)
print('sar img shape:', img_sar.shape)
plt.subplot(1,2,2)
plt.imshow(img_sar, cmap='gray')
plt.title('original sar img')
plt.show()
