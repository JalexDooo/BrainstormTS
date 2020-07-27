import torch
import torch.nn as nn
import math

from .BasicModule import *


"""
Main class: class Yi_Ding()
Note: This model have no batch-normalization/group-normalization function() refferring to that paper.
Error: According to Fig.1 in the paper, the channel of TD1/TD2/TD3 should be 5/6/8 after concatenation.
       But according to Tab.3 in the paper, the channel of TD1/TD2/TD3 is 1/1/1 .
Other idea: Does lots of residual block work without normalization function? 
"""


class Proposed_DenseBlock(nn.Module):
    def __init__(self, in_data, k):
        super(Proposed_DenseBlock, self).__init__()
        self.k = k
        self.ds4 = nn.Sequential(
            nn.Conv3d(in_data, in_data, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_data, in_data, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_data, in_data, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_data, in_data, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.ds5 = nn.Sequential(
            nn.Conv3d(in_data, in_data, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_data, in_data, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_data, in_data, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_data, in_data, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_data, in_data, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.ds7 = nn.Sequential(
            nn.Conv3d(in_data, in_data, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_data, in_data, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_data, in_data, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_data, in_data, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_data, in_data, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_data, in_data, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_data, in_data, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.ds10 = nn.Sequential(
            nn.Conv3d(in_data, in_data, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_data, in_data, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_data, in_data, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_data, in_data, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_data, in_data, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_data, in_data, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_data, in_data, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_data, in_data, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_data, in_data, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_data, in_data, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.ds = nn.Sequential(
            nn.Conv3d(in_data, in_data, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if self.k == 4:
            x1 = self.ds(x)
            x2 = self.ds(x1 + x)
            x3 = self.ds(x2 + x1 + x)
            x1 = self.ds(x3 + x2 + x1)
            return x1+x
        elif self.k == 5:
            x1 = self.ds(x)
            x2 = self.ds(x1 + x)
            x3 = self.ds(x2 + x1 + x)
            x1 = self.ds(x3 + x2 + x1)
            x2 = self.ds(x3 + x2 + x1)
            return x2+x
        elif self.k == 7:
            x1 = self.ds(x)
            x2 = self.ds(x1 + x)
            x3 = self.ds(x2 + x1 + x)
            x1 = self.ds(x3 + x2 + x1)
            x2 = self.ds(x3 + x2 + x1)
            x3 = self.ds(x3 + x2 + x1)
            x1 = self.ds(x3 + x2 + x1)
            return x+x1
        else:
            x1 = self.ds(x)
            x2 = self.ds(x1 + x)
            x3 = self.ds(x2 + x1 + x)
            x1 = self.ds(x3 + x2 + x1)
            x2 = self.ds(x3 + x2 + x1)
            x3 = self.ds(x3 + x2 + x1)
            x1 = self.ds(x3 + x2 + x1)
            x2 = self.ds(x3 + x2 + x1)
            x3 = self.ds(x3 + x2 + x1)
            x1 = self.ds(x3 + x2 + x1)
            return x+x1


class Multi_Path_Adaptive_Fusion_Dense_Block(nn.Module):
    def __init__(self, k1, k2):
        super(Multi_Path_Adaptive_Fusion_Dense_Block, self).__init__()
        self.db1 = Proposed_DenseBlock(16, k1)
        self.db2 = Proposed_DenseBlock(16, k2)
        self.conv = nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1)
        self.tranpose_conv = nn.ConvTranspose3d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1)

    def forward(self, x1, x2):
        x1_ = self.db1(x1)
        x1_ = self.conv(x1_)
        x2_ = self.conv(x2)
        x2_ = self.tranpose_conv(x2_)
        x = x1_ + x2_
        x = self.db2(x)
        return x



class Yi_Ding(nn.Module):
    def __init__(self, in_data=4, out_data=5):
        super(Yi_Ding, self).__init__()

        self.conv1 = nn.Conv3d(in_data, 16, kernel_size=7, stride=2, padding=3)
        self.db1 = Proposed_DenseBlock(16, k=4)
        self.td = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=1, stride=1, padding=0),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.db2 = Proposed_DenseBlock(16, 5)
        self.db3 = Proposed_DenseBlock(16, 7)
        self.db4 = Proposed_DenseBlock(16, 10)

        self.mp1 = Multi_Path_Adaptive_Fusion_Dense_Block(10, 7)
        self.mp2 = Multi_Path_Adaptive_Fusion_Dense_Block(7, 5)
        self.mp3 = Multi_Path_Adaptive_Fusion_Dense_Block(5, 4)

        self.conv2 = nn.ConvTranspose3d(16, out_data, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1)

    def forward(self, x):
        x = self.conv1(x)

        x1 = self.db1(x)
        x = self.td(x1)

        x2 = self.db2(x)
        x = self.td(x2)

        x3 = self.db3(x)
        x = self.td(x3)

        x4 = self.db4(x)
        # print('x1: {}, x2: {}, x3: {}, x4: {}'.format(x1.shape, x2.shape, x3.shape, x4.shape))

        x = self.mp1(x3, x4)
        x = self.mp2(x2, x)
        x = self.mp3(x1, x)
        x = self.conv2(x)
        # print(x.shape)
        return x


if __name__ == '__main__':
    model = Yi_Ding(in_data=1, out_data=2)
    data = torch.randn(2, 1, 16, 16, 64)
    out = model(data)
