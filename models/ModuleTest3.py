import torch as t
import torch.nn as nn
import math

from .BasicModule import *

class ModuleTest3(nn.Module):
    def __init__(self, in_data=4, out_data=5):
        super(ModuleTest3, self).__init__()
        kn = [16, 32, 64, 128]

        self.layer1_conv1 = DilationConvBlock(in_data, kn[0])
        self.layer1_conv2 = ConvBlockWithKernel3(kn[0], kn[1])
        self.layer1_conv3 = ConvBlockWithKernel3(kn[1], kn[2])
        self.layer1_conv4 = ConvBlockWithKernel3(kn[2], kn[3])

        self.layer2_conv1 = DilationConvBlock(kn[0], kn[1])
        self.layer2_conv2 = ConvBlockWithKernel3(kn[1], kn[2])
        self.layer2_conv3 = ConvBlockWithKernel3(kn[2], kn[3])

        self.layer3_conv1 = DilationConvBlock(kn[1], kn[2])
        self.layer3_conv2 = ConvBlockWithKernel3(kn[2], kn[3])

        self.layer4_conv1 = DilationConvBlock(kn[2], kn[3])


        self.layer34_1_transconv = TransConvBlock(kn[3], kn[2], 2)
        self.layer34_2_transconv = TransConvBlock(kn[2], kn[0], 2)

        self.layer34_1_conv = ConvBlockWithKernel3(kn[3], kn[3])
        self.layer34_2_conv = ConvBlockWithKernel3(kn[2], kn[2])
        self.layer34_3_conv = ConvBlockWithKernel3(kn[0], kn[0])

        self.layer2_1_transconv = TransConvBlock(kn[3], kn[2], 2)
        self.layer2_2_transconv = TransConvBlock(kn[2], kn[0], 2)

        self.layer1_transconv = TransConvBlock(kn[3], kn[0], 2)
        self.layer2_transconv = TransConvBlock(kn[3], kn[0], 4)
        self.layer3_transconv = TransConvBlock(kn[3], kn[0], 8)
        self.layer4_transconv = TransConvBlock(kn[3], kn[0], 8)


        self.out_conv1 = ConvBlockWithKernel3(48, 48)
        self.out_tconv2 = TransConvBlock(48, 12, 4)
        self.out_conv3 = ConvBlockWithKernel3(12, 12)
        self.out_conv4 = ConvBlockWithKernel3(12, out_data)


        self.pre_out = TransConvBlock(kn[0]*4, kn[0], 2)
        self.out = ConvBlockWithKernel3(kn[0], out_data)

        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x1 = self.layer1_conv1(x)
        # print('x1.shape: ', x1.shape)
        x2 = self.layer2_conv1(x1)
        # print('x2.shape: ', x2.shape)
        x3 = self.layer3_conv1(x2)
        # print('x3.shape: ', x3.shape)
        x4 = self.layer4_conv1(x3)
        # print('x4.shape: ', x4.shape)
        x1 = self.max_pool(x1)
        x2 = self.max_pool(x2)
        x3 = self.max_pool(x3)
        # x4 = self.layer4_pool(x4)
        # print('pool1.shape: ', x1.shape)
        # print('pool2.shape: ', x2.shape)
        # print('pool3.shape: ', x3.shape)
        # print('pool4.shape: ', x4.shape)
        '''
        x1.shape:  torch.Size([8, 16, 72, 96, 48])
        x2.shape:  torch.Size([8, 32, 36, 48, 24])
        x3.shape:  torch.Size([8, 64, 18, 24, 12])
        x4.shape:  torch.Size([8, 128, 9, 12, 6])
        pool1.shape:  torch.Size([8, 16, 36, 48, 24])
        pool2.shape:  torch.Size([8, 32, 18, 24, 12])
        pool3.shape:  torch.Size([8, 64, 9, 12, 6])
        pool4.shape:  torch.Size([8, 128, 4, 6, 3])
        '''
        x1 = self.layer1_conv2(x1)
        x1 = self.layer1_conv3(x1)
        x1 = self.layer1_conv4(x1)

        x2 = self.layer2_conv2(x2)
        x2 = self.layer2_conv3(x2)

        x3 = self.layer3_conv2(x3)

        # print('x1.shape: ', x1.shape)
        # print('x2.shape: ', x2.shape)
        # print('x3.shape: ', x3.shape)
        # print('x4.shape: ', x4.shape)


        '''
        x1.shape:  torch.Size([2, 128, 36, 48, 24])
        x2.shape:  torch.Size([2, 128, 18, 24, 12])
        x3.shape:  torch.Size([2, 128, 9, 12, 6])
        x4.shape:  torch.Size([2, 128, 9, 12, 6])
        '''

        x3 = self.layer34_1_transconv(x3)
        x4 = self.layer34_1_transconv(x4)
        t2 = t.cat([x3, x4], dim=1)
        # t2.shape:  torch.Size([2, 128, 18, 24, 12])
        t2 = self.layer34_1_conv(t2)
        t2 = self.layer34_1_transconv(t2)
        t2 = self.layer34_2_conv(t2)
        t2 = self.layer34_2_transconv(t2)
        t2 = self.layer34_3_conv(t2)
        # t2.shape: torch.Size([2, 16, 72, 96, 48])

        x2 = self.layer2_1_transconv(x2)
        x2 = self.layer34_2_conv(x2)
        x2 = self.layer2_2_transconv(x2)
        x2 = self.layer34_3_conv(x2)

        x1 = self.layer1_transconv(x1)
        x1 = self.layer34_3_conv(x1)

        # print('t2.shape: ', t2.shape)
        # print('x2.shape: ', x2.shape)
        # print('x1.shape: ', x1.shape)
        '''
        t2.shape: torch.Size([2, 16, 72, 96, 48])
        x2.shape: torch.Size([2, 16, 72, 96, 48])
        x1.shape: torch.Size([2, 16, 72, 96, 48])
        '''


        t1 = t.cat([x1, x2], dim=1)
        x = t.cat([t1, t2], dim=1)
        # print('x.shape: ', x.shape)
        '''
        x.shape:  torch.Size([8, 48, 72, 96, 48])
        '''

        x = self.out_conv1(x)
        x = self.max_pool(x)
        # print('x.shape: ', x.shape)
        '''
        x.shape:  torch.Size([8, 24, 36, 48, 24])
        '''

        x = self.out_tconv2(x)
        x = self.out_conv3(x)
        x = self.out_conv4(x)


        return x

