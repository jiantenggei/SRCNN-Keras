import os
import cv2
import numpy as np
from PIL import Image

def load_train(image_size=33, stride=33, scale=3,dirname=r'dataset\train'):
    dir_list = os.listdir(dirname)
    images = [cv2.cvtColor(cv2.imread(os.path.join(dirname,img)),cv2.COLOR_BGR2GRAY) for img in dir_list]
    #==========================
    #这里判断采样步长 是否能被整除
    #=========================
    images = [img[0:img.shape[0]-np.remainder(img.shape[0],scale),0:img.shape[1]-np.remainder(img.shape[1],scale)] for img in images]

    trains = images.copy()
    labels = images.copy()
    #========================================
    #对train image 进行方法缩小 产生不清晰的图像
    #========================================
    trains = [cv2.resize(img, None, fx=1/scale, fy=1/scale, interpolation=cv2.INTER_CUBIC) for img in trains]
    trains = [cv2.resize(img, None, fx=scale/1, fy=scale/1, interpolation=cv2.INTER_CUBIC) for img in trains]

    sub_trains = []
    sub_labels = []
    
    #========================================
    #通过采样形成标签 和训练数据，
    # 一张 图片 通过采样，可以分成很多个图像块，作为训练数据，丰富样本
    #========================================
    for train, label in zip(trains, labels):
        v, h = train.shape
        print(train.shape)
        for x in range(0,v-image_size+1,stride):
            for y in range(0,h-image_size+1,stride):
                sub_train = train[x:x+image_size,y:y+image_size]
                sub_label = label[x:x+image_size,y:y+image_size]
                sub_train = sub_train.reshape(image_size,image_size,1)
                sub_label = sub_label.reshape(image_size,image_size,1)
                sub_trains.append(sub_train)
                sub_labels.append(sub_label)
    #========================================
    #编码为numpy array
    #========================================
    sub_trains = np.array(sub_trains)
    sub_labels = np.array(sub_labels)
    return sub_trains, sub_labels

def load_test(scale=3,dirname=r'dataset\test'):
    #========================================
    # 生成测试数据的方式与训练数据相同
    # pre_tests 是用来保存缩小后的图片
    #========================================
    dir_list = os.listdir(dirname)
    images = [cv2.cvtColor(cv2.imread(os.path.join(dirname,img)),cv2.COLOR_BGR2GRAY) for img in dir_list]
    images = [img[0:img.shape[0]-np.remainder(img.shape[0],scale),0:img.shape[1]-np.remainder(img.shape[1],scale)] for img in images]

    tests = images.copy()
    labels = images.copy()
    
    pre_tests = [cv2.resize(img, None, fx=1/scale, fy=1/scale, interpolation=cv2.INTER_CUBIC) for img in tests]
    tests = [cv2.resize(img, None, fx=scale/1, fy=scale/1, interpolation=cv2.INTER_CUBIC) for img in pre_tests]
    
    pre_tests = [img.reshape(img.shape[0],img.shape[1],1) for img in pre_tests]
    tests = [img.reshape(img.shape[0],img.shape[1],1) for img in tests]
    labels = [img.reshape(img.shape[0],img.shape[1],1) for img in labels]

    return pre_tests, tests, labels

#========================================
# 下面函数用来计算重构前后的图片指标
#========================================
def mse(y, t):
    return np.mean(np.square(y - t))

def psnr(y, t):
    return 20 * np.log10(255) - 10 * np.log10(mse(y, t))

def ssim(x, y):
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    var_x = np.var(x)
    var_y = np.var(y)
    cov = np.mean((x - mu_x) * (y - mu_y))
    c1 = np.square(0.01 * 255)
    c2 = np.square(0.03 * 255)
    return ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))



