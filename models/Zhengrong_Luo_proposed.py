import torch
import torch.nn as nn
import math

from .BasicModule import *


"""
Main class: class Zhengrong_Luo():
Notes: 
    The deep of network: We use [32, 32, 32, 32, 32].
    The down-sampling operate is max-pooling, convolution with stride of 2 or other? We use max-pooling.
    The last input dim need to be divided by 64.
"""


class BasicConv(nn.Module):
    def __init__(self, in_data, out_data):
        super(BasicConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_data, out_data, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_data),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.conv(x)
        return x


class HDC_module(nn.Module):
    """
    Notes: The in_data -> out_data is confusing! We put it in self.output_conv().
    """
    def __init__(self, in_data, out_data):
        super(HDC_module, self).__init__()
        self.in_out_conv = nn.Sequential(
            nn.Conv3d(in_data, in_data, kernel_size=(1, 1, 1), stride=1, padding=0),
            nn.BatchNorm3d(in_data),
            nn.ReLU(inplace=True)
        )
        self.inner_conv = nn.Sequential(
            nn.Conv3d(in_data, in_data, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0)),
            nn.BatchNorm3d(in_data),
            nn.ReLU(inplace=True)
        )
        self.output_conv = nn.Sequential(
            nn.Conv3d(in_data, out_data, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.BatchNorm3d(out_data),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        channel = x.shape[-1]//4
        x_in = self.in_out_conv(x)
        x_in1 = x_in[:, :, :, :, :channel]
        x_in2 = self.inner_conv(x_in[:, :, :, :, channel:channel*2])
        x_in3 = self.inner_conv(x_in2+x_in[:, :, :, :, channel*2:channel*3])
        x_in4 = self.inner_conv(x_in3+x_in[:, :, :, :, channel*3:])

        x_cat = torch.cat([x_in1, x_in2], dim=-1)
        x_cat = torch.cat([x_cat, x_in3], dim=-1)
        x_cat = torch.cat([x_cat, x_in4], dim=-1)

        x_ = self.in_out_conv(x_cat)
        x_ = self.output_conv(x_)
        x = x + x_
        x = x.cuda()

        return x


class Zhengrong_Luo(nn.Module):
    """
    HDC-Net++ 3D Convolution Version.
    """
    def __init__(self, in_data=4, out_data=5):
        super(Zhengrong_Luo, self).__init__()
        deep = [32, 32, 32, 32, 32]
        self.conv1 = nn.Sequential(
            BasicConv(in_data, deep[0]),
            # nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.hdc1 = nn.Sequential(
            HDC_module(deep[0], deep[1]),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.hdc2 = nn.Sequential(
            HDC_module(deep[1], deep[2]),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.hdc3 = nn.Sequential(
            HDC_module(deep[2], deep[3]),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.hdc4 = nn.Sequential(
            HDC_module(deep[3], deep[4]),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.trans1 = UpBlock(deep[4], deep[3])
        self.trans2 = UpBlock(deep[3], deep[2])
        self.trans3 = UpBlock(deep[2], deep[1])
        self.trans4 = UpBlock(deep[1], deep[0])

        self.hdc = HDC_module(32, 32)

        self.conv2 = nn.Sequential(
            BasicConv(deep[0], out_data),
            # nn.MaxPool3d(kernel_size=2, stride=2)
        )

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x_ = self.conv1(x)
        x1 = self.hdc1(x_)
        x2 = self.hdc2(x1)
        x3 = self.hdc3(x2)
        x4 = self.hdc4(x3)
        # print('x_.shape: ', x_.shape)
        # print('x1.shape: ', x1.shape)
        # print('x2.shape: ', x2.shape)
        # print('x3.shape: ', x3.shape)
        # print('x4.shape: ', x4.shape)
        x = self.trans1(x4, x3)
        x = self.hdc(x)
        x = self.trans2(x, x2)
        x = self.hdc(x)
        x = self.trans3(x, x1)
        x = self.hdc(x)
        x = self.trans4(x, x_)
        x = self.hdc(x)
        x = self.conv2(x)

        return x


if __name__ == '__main__':
    model = Zhengrong_Luo(in_data=1, out_data=2)
    data = torch.randn(2, 1, 16, 16, 64)
    out = model(data)