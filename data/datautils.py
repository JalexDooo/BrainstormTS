import os
import sys
import glob
import numpy as np
import pandas as pd
import nibabel as nib

import matplotlib.pylab as plt
import matplotlib.cm as cm
import matplotlib.image as Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import scipy.misc

def load_hgg_lgg_files(path):
    """
        加载HGG和LGG所有文件，以每个文件夹作为一个训练数据自己。
    """
    return glob.glob(path+'/HGG/*') + glob.glob(path+'/LGG/*')

def load_val_file(path):
    return glob.glob(path+'/*')

def load_nii_to_array(path):
    """
        将.nii格式加载成图像数据。
    """
    image = nib.load(path)
    image = image.get_data()

    image = np.transpose(image, [2, 1, 0])
    # image = np.array(image) ###
    return image

def make_image_label(path):
    """
        将一个文件夹下的数据加载为一个样本，顺序：flair,t1,t1ce,t2
    """
    pathes = glob.glob(path+'/*.nii.gz')
    image = []
    seg = None

    for p in pathes:
        if 'flair.nii' in p:
            flair = load_nii_to_array(p)
        elif 't2.nii' in p:
            t2 = load_nii_to_array(p)
        elif 't1.nii' in p:
            t1 = load_nii_to_array(p)
        elif 't1ce.nii' in p:
            t1ce = load_nii_to_array(p)
        else:
            seg = load_nii_to_array(p)
    image.append(flair)
    image.append(t1)
    image.append(t1ce)
    image.append(t2)

    label = seg
    return image, label

def get_box(image, margin):
    """
        抠图，将图像四周无像素区域扣掉，margin为预留区域参数，如：margin=3，表示有像素点周围预留三个像素点距离。
    """
    shape = image.shape
    nonindex = np.nonzero(image) # 返回的是3个数组，分别对应三个维度的下标。

    margin = [margin] * len(shape)

    index_min = []
    index_max = []

    for i in range(len(shape)):
        index_min.append(nonindex[i].min())
        index_max.append(nonindex[i].max())
    
    # 扩大margin个区域
    for i in range(len(shape)):
        index_min[i] = max(index_min[i] - margin[i], 0)
        index_max[i] = min(index_max[i] + margin[i], shape[i]-1)
    
    # print(index_min)
    # print(index_max)
    return index_min, index_max

def make_box(image, index_min, index_max, data_box):
    """
        抠图，使用get_box()获得的下标。
    """
    shape = image.shape

    for i in range(len(shape)):

        # print('before index[%s]: '%i, index_min[i], index_max[i])

        # 按照data_box扩大或缩小box。
        mid = (index_min[i] + index_max[i])/2
        index_min[i] = mid - data_box[i]/2
        index_max[i] = mid + data_box[i]/2
        
        flag = index_max[i] - shape[i]
        if flag > 0:
            index_max[i] = index_max[i] - flag
            index_min[i] = index_min[i] - flag
        
        flag = index_min[i]
        if flag < 0:
            index_max[i] = index_max[i] - flag
            index_min[i] = index_min[i] - flag
        
        # print('index[%s]: '%i, index_min[i], index_max[i])

        if index_max[i] - index_min[i] != data_box[i]:
            index_max[i] = index_min[i] + data_box[i]
    
        index_max[i] = int(index_max[i])
        index_min[i] = int(index_min[i])

        # print('after index[%s]: '%i, index_min[i], index_max[i])
    return index_min, index_max
    
def crop_with_box(image, index_min, index_max):
    """
        按照box分割图像。
    """
    # return image[np.ix_(range(index_min[0], index_max[0]), range(index_min[1], index_max[1]), range(index_min[2], index_max[2]))]
    x = index_max[0] - index_min[0] - image.shape[0]
    y = index_max[1] - index_min[1] - image.shape[1]
    z = index_max[2] - index_min[2] - image.shape[2]
    img = image
    img1 = image
    img2 = image

    if x > 0:
        img = np.zeros((image.shape[0]+x, image.shape[1], image.shape[2]))
        img[x//2:image.shape[0]+x//2, :, :] = image[:, :, :]
        img1 = img

    if y > 0:
        img = np.zeros((img1.shape[0], img1.shape[1]+y, img1.shape[2]))
        img[:, y//2:image.shape[1]+y//2, :] = img1[:, :, :]
        img2 = img

    if z > 0:
        img = np.zeros((img2.shape[0], img2.shape[1], img2.shape[2]+z))
        img[:, :, z//2:image.shape[2]+z//2] = img2[:, :, :]


    # print('----')
    # print(image.shape)
    # print(img.shape)
    # print(index_min, index_max)
    # print('----')

    return img[np.ix_(range(index_min[0], index_max[0]), range(index_min[1], index_max[1]), range(index_min[2], index_max[2]))]



def normalization(image):
    """
        图像标准化。Z-Score标准化，标准差标准化。使得结果符合正态分布。
    """
    img = image[image>0]
    image = (image - img.mean()) / img.std()
    return image

def get_WT_labels(image):
    return (image == 1) * 1.0 + (image == 2) * 1.0 + (image == 4) * 1.0

def get_TC_labels(image):
    return (image == 1) * 1.0 + (image == 4) * 1.0

def get_ET_labels(image):
    return (image == 4) * 1.0

def get_NCR_NET_label(image):
    """
    ET: enhancing tumor
    For ET task.
    :param image:
    :return:
    """
    return (image == 1)*1.0

def get_precise_labels(image):
    return image*1.0

########################### 动脉瘤处理
def load_aneu_image_path(path):
    return glob.glob(path+'/1/*') + glob.glob(path+'/2/*') + glob.glob(path+'/3/*')

## WT: label==2, ET: label==4, TC: label==1
# image = load_nii_to_array('/Users/juntysun/Downloads/Create/BrainstormTS/brats2019_val_moduletest5_random/BraTS19_UAB_3448_1.nii.gz')
# print((image==1).sum())
# print((image==2).sum())
# print((image==4).sum())
# 1247
# 22514
# 12776
