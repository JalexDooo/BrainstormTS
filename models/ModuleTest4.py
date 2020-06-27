import torch as t
import torch.nn as nn
import math

from .BasicModule import *

class mytc(nn.Module):
    def __init__(self, ch1, ch2, kn):
        super(mytc, self).__init__()
        self.up = TransConvBlock(ch1, ch2, 2)
        self.down = ConvBlockWithKernel3(ch2+kn, ch2)

    def forward(self, x, origin):
        x = self.up(x)
        x = t.cat([x, origin], dim=1)
        x = self.down(x)
        return x

class ModuleTest4(nn.Module):
    def __init__(self, in_data=4, out_data=5):
        super(ModuleTest4, self).__init__()
        kn = [32, 64, 128, 256]

        # pathway-1
        self.in1_model = nn.Sequential(
            DilationConvBlock(in_data, kn[0]),
            ConvBlockWithKernel3(kn[0], kn[0])
        )

        self.layer1_1 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            ConvBlockWithKernel3(kn[0], kn[1])
        )
        self.layer1_2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            ConvBlockWithKernel3(kn[1], kn[2])
        )
        self.layer1_3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            ConvBlockWithKernel3(kn[2], kn[3])
        )

        #pathway-2
        self.in2_model = nn.Sequential(
            DilationConvBlock(kn[0], kn[1]),
            ConvBlockWithKernel3(kn[1], kn[1])
        )
        self.layer2_1 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            ConvBlockWithKernel3(kn[1], kn[2])
        )
        self.layer2_2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            ConvBlockWithKernel3(kn[2], kn[3])
        )

        #pathway-3
        self.in3_model = nn.Sequential(
            DilationConvBlock(kn[1], kn[2]),
            ConvBlockWithKernel3(kn[2], kn[2])
        )
        self.layer3_1 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            ConvBlockWithKernel3(kn[2], kn[3])
        )

        #pathway-4
        self.in4_model = nn.Sequential(
            DilationConvBlock(kn[2], kn[3]),
            ConvBlockWithKernel3(kn[3], kn[3])
        )

        self.tc1 = nn.Sequential(
            TransConvBlock(kn[3]*4, kn[3]*2, 2),
            ConvBlockWithKernel3(kn[3]*2, kn[3]*2)
        )
        self.tc2 = nn.Sequential(
            TransConvBlock(kn[3]*2, kn[3], 2),
            ConvBlockWithKernel3(kn[3], kn[3])
        )
        self.tc3 = nn.Sequential(
            TransConvBlock(kn[3], kn[2], 2),
            ConvBlockWithKernel3(kn[2], kn[2])
        )
        self.tc4 = nn.Sequential(
            TransConvBlock(kn[2], kn[1], 2),
            ConvBlockWithKernel3(kn[1], kn[1])
        )

        self.mytc1 = mytc(128 * 4, 128 * 2, 64)
        self.mytc2 = mytc(128 * 2, 128, 32)
        self.mytc3 = mytc(128, 64, 16)
        self.mytc4 = mytc(64, 16, 4)



        self.out_model = ConvBlockWithKernel3(kn[0], out_data)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, ind):
        x1_ = self.in1_model(ind)
        # print('x1_.shape: ', x1_.shape)
        x2_ = self.in2_model(x1_)
        # print('x2_.shape: ', x2_.shape)
        x3_ = self.in3_model(x2_)
        # print('x3_.shape: ', x3_.shape)
        x4 = self.in4_model(x3_)
        # print('x4.shape: ', x4.shape)
        '''
        x1_.shape:  torch.Size([1, 16, 72, 96, 48])
        x2_.shape:  torch.Size([1, 32, 36, 48, 24])
        x3_.shape:  torch.Size([1, 64, 18, 24, 12])
        x4.shape:  torch.Size([1, 128, 9, 12, 6])
        '''

        # pathway-1
        x1 = self.layer1_1(x1_)
        x1 = self.layer1_2(x1)
        x1 = self.layer1_3(x1)

        # pathway-2
        x2 = self.layer2_1(x2_)
        x2 = self.layer2_2(x2)

        # pathway-3
        x3 = self.layer3_1(x3_)

        # print('x1.shape: ', x1.shape)
        # print('x2.shape: ', x2.shape)
        # print('x3.shape: ', x3.shape)
        # print('x4.shape: ', x4.shape)
        '''
        x1, x2, x3, x4: torch.Size([1, 128, 9, 12, 6])
        '''

        t1 = t.cat([x1, x2], dim=1)
        t2 = t.cat([x3, x4], dim=1)
        x = t.cat([t1, t2], dim=1)

        x = self.mytc1(x, x3_)
        x = self.mytc2(x, x2_)
        x = self.mytc3(x, x1_)
        x = self.mytc4(x, ind)

        # x = self.tc1(x)
        # # print('x1.shape: ', x.shape)
        # x = self.tc2(x)
        # # print('x2.shape: ', x.shape)
        # x = self.tc3(x)
        # # print('x3.shape: ', x.shape)
        # x = self.tc4(x)
        # # print('x4.shape: ', x.shape)
        '''
        x1.shape:  torch.Size([1, 256, 18, 24, 12])
        x2.shape:  torch.Size([1, 128, 36, 48, 24])
        x3.shape:  torch.Size([1, 64, 72, 96, 48])
        x4.shape:  torch.Size([1, 32, 144, 192, 96])
        '''
        # print('x.shape: ', x.shape)
        '''
        torch.Size([1, 32, 144, 192, 96])
        '''

        x = self.out_model(x)
        # return

        return x

