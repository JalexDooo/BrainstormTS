import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import SimpleITK as sitk


def netSize(net):
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    return k


def dice(predict, target):
    """

    :param predict: 4D Long Tensor Batch_Size * 16(volume_size) * height * weight
    :param target:  4D Long Tensor Batch_Size * 16(volume_size) * height * weight
    :return:
    """
    smooth = 0.00000001
    batch_num = target.shape[0]
    target = target.view(batch_num, -1)
    predict = predict.view(batch_num, -1)
    intersection = float((target * predict).sum())

    return (2.0 * intersection + smooth) / (float(predict.sum())
                                            + float(target.sum()) + smooth)


def score(predict, target):
    e = 0.000000001
    # print('score begin:')

    batch_num = target.shape[0]
    target = target.view(batch_num, -1)
    predict = predict.view(batch_num, -1)
    # print('target: ', target.shape)
    # print('predict: ', predict.shape)
    t1 = float((1.0*(predict==target)).sum())+e
    t2 = float(len(target[0])*batch_num)+e
    # print('t1: ', t1)
    # print('t2: ', t2)

    return t1 / t2


def save_array_as_nifty_volume(data, filename, reference_name = None):
    """
    save a numpy array as nifty image
    inputs:
        data: a numpy array with shape [Depth, Height, Width]
        filename: the ouput file name
        reference_name: file name of the reference image of which affine and header are used
    outputs: None
    """
    img = sitk.GetImageFromArray(data)
    if(reference_name is not None):
        img_ref = sitk.ReadImage(reference_name)
        img.CopyInformation(img_ref)
    sitk.WriteImage(img, filename)


def make_box(image, index_min, index_max, data_box):
    """
        抠图，使用get_box()获得的下标。
    """
    shape = image.shape

    for i in range(len(shape)):

        # print('before index[%s]: '%i, index_min[i], index_max[i])

        # 按照data_box扩大或缩小box。
        mid = (index_min[i] + index_max[i]) / 2
        index_min[i] = mid - data_box[i] / 2
        index_max[i] = mid + data_box[i] / 2

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
    return image[
        np.ix_(range(index_min[0], index_max[0]), range(index_min[1], index_max[1]), range(index_min[2], index_max[2]))]


def out_precessing(label):
    tmp = np.asarray(label)
    if (tmp==4).sum() <= 500:
        tmp = (tmp == 4)*1 + (tmp == 1)*1 + (tmp==2)*2 + (tmp==3)*3
    return tmp

#
# image = [[0, 1, 1, 2, 2, 3, 3, 4, 4]]
# print(out_precessing(image))
