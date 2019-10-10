import torch
import torch.nn as nn


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

    batch_num = target.shape[0]
    target = target.view(batch_num, -1)
    predict = predict.view(batch_num, -1)

    return (1.0*(predict==target)).sum() / float(len(target[0])*batch_num)