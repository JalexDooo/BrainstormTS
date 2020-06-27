import torch as t
import torch.nn as nn
import math

from .BasicModule import *


class ResUnet(nn.Module):
    def __init__(self, in_data=4, out_data=5):
        super(ResUnet, self).__init__()
        kn = [32, 64, 128, 256]

        self.in_model = NewConvDown(in_data, kn[0], stride=1)

        self.layer1 = nn.Sequential(
            NewResBlock(kn[0], kn[0]),
            NewResBlock(kn[0], kn[0]),
            NewConvDown(kn[0], kn[1], stride=2)
        )
        self.layer2 = nn.Sequential(
            NewResBlock(kn[1], kn[1]),
            NewResBlock(kn[1], kn[1]),
            NewConvDown(kn[1], kn[2], stride=2)
        )
        self.layer3 = nn.Sequential(
            NewResBlock(kn[2], kn[2]),
            NewResBlock(kn[2], kn[2]),
            NewConvDown(kn[2], kn[3], stride=2)
        )

        self.bottom = nn.Sequential(
            NewResBlock(kn[3], kn[3]),
            NewResBlock(kn[3], kn[3]),
            NewResBlock(kn[3], kn[3]),
            NewConvUp(kn[3], kn[2], stride=2)
        )

        self.llayer3 = nn.Sequential(
            NewResBlock(kn[2]*2, kn[2]),
            NewResBlock(kn[2], kn[2]),
            NewConvUp(kn[2], kn[1], stride=2)
        )

        self.llayer2 = nn.Sequential(
            NewResBlock(kn[1]*2, kn[1]),
            NewResBlock(kn[1], kn[1]),
            NewConvUp(kn[1], kn[0], stride=2)
        )

        self.out = OutConv(kn[0]*2, out_data)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, ind):
        x_ = self.in_model(ind)
        x1 = self.layer1(x_)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        bottom = self.bottom(x3)

        # print('shape: bottom: ', bottom.shape)
        # print('shape: x3: ', x3.shape)
        # print('shape: x2: ', x2.shape)
        # print('shape: x1: ', x1.shape)

        x = t.cat([bottom, x2], dim=1)
        x = self.llayer3(x)
        x = t.cat([x, x1], dim=1)
        x = self.llayer2(x)
        x = t.cat([x, x_], dim=1)

        out = self.out(x)


        return out

