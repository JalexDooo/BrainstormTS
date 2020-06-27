import torch as t
import torch.nn as nn
import math

from .BasicModule import *

class ModuleTest1(nn.Module):
    def __init__(self, in_data=4, out_data=5):
        super(ModuleTest1, self).__init__()
        kn = [16, 32, 60]

        self.layer1_conv1 = DilationConvBlock(in_data, 16)
        self.layer1_conv2 = ConvBlockWithKernel3(16, kn[0])
        self.layer1_conv3 = ConvBlockWithKernel3(kn[0], kn[1])
        self.layer1_conv4 = ConvBlockWithKernel3(kn[1], kn[2])

        self.layer2_conv1 = DilationConvBlock(kn[0], kn[0])
        self.layer2_conv2 = ConvBlockWithKernel3(kn[0], kn[1])
        self.layer2_conv3 = ConvBlockWithKernel3(kn[1], kn[2])

        self.layer3_conv1 = DilationConvBlock(kn[1], kn[1])
        self.layer3_conv2 = ConvBlockWithKernel3(kn[1], kn[2])

        self.layer4_conv1 = DilationConvBlock(kn[2], kn[2])

        self.layer1_transconv = TransConvBlock(kn[2], kn[0], 2)
        self.layer2_transconv = TransConvBlock(kn[2], kn[0], 4)
        self.layer3_transconv = TransConvBlock(kn[2], kn[0], 8)
        self.layer4_transconv = TransConvBlock(kn[2], kn[0], 16)

        self.pre_out = ConvBlockWithKernel3(kn[0]*4, kn[0])
        self.out = ConvBlockWithKernel3(kn[0], out_data)

    def forward(self, x):
        x1 = self.layer1_conv1(x)
        x1 = self.layer1_conv2(x1)
        x2 = self.layer2_conv1(x1)
        x2 = self.layer2_conv2(x2)
        x3 = self.layer3_conv1(x2)
        x3 = self.layer3_conv2(x3)
        x4 = self.layer4_conv1(x3)

        x1 = self.layer1_conv3(x1)
        x1 = self.layer1_conv4(x1)
        x2 = self.layer2_conv3(x2)

        x1 = self.layer1_transconv(x1)
        x2 = self.layer2_transconv(x2)
        x3 = self.layer3_transconv(x3)
        x4 = self.layer4_transconv(x4)

        t1 = t.cat([x1, x2], dim=1)
        t2 = t.cat([x3, x4], dim=1)
        x = t.cat([t1, t2], dim=1)

        x = self.pre_out(x)
        x = self.out(x)
        return x

"""

x1: [2, , 72, 96, 48]  *2
x2: [2, , 36, 48, 24]  *4
x3: [2, , 18, 24, 12]  *8
x4: [2, , 9, 12, 6]    *16

trans: [2, , 144, 192, 96]


"""