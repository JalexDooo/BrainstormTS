import torch
import torch.nn as nn
import math

from .BasicModule import *

class UNet3D_Aneu(nn.Module):
    """
    LReLU
    """
    def __init__(self, in_data=1, out_data=2, degree=16):
        super(UNet3D_Aneu, self).__init__()

        drop = []
        for i in range(5):
            drop.append((2 ** i) * degree)
        print('UNet3D drop: ', drop)  # [16, 32, 64, 128, 256]

        self.downLayer1 = LConvBlock(in_data, drop[0])
        self.downLayer2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            LConvBlock(drop[0], drop[1])
        )
        self.downLayer3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            LConvBlock(drop[1], drop[2])
        )
        self.downLayer4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            LConvBlock(drop[2], drop[3])
        )
        self.bottomLayer = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            LConvBlock(drop[3], drop[4])
        )

        self.upLayer1 = UpBlock(drop[4], drop[3])
        self.upLayer2 = UpBlock(drop[3], drop[2])
        self.upLayer3 = UpBlock(drop[2], drop[1])
        self.upLayer4 = UpBlock(drop[1], drop[0])

        self.outLayer = nn.Conv3d(drop[0], out_data, kernel_size=3, stride=1, padding=1)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x1 = self.downLayer1(x)
        x2 = self.downLayer2(x1)
        x3 = self.downLayer3(x2)
        x4 = self.downLayer4(x3)
        bottom = self.bottomLayer(x4)
        x = self.upLayer1(bottom, x4)
        x = self.upLayer2(x4, x3)
        x = self.upLayer3(x, x2)
        x = self.upLayer4(x, x1)
        x = self.outLayer(x)
        return x

