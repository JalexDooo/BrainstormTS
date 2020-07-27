import torch
import torch.nn as nn
import math

from .BasicModule import *


"""
Main class: class Liangliang_Liu()
Notes: The channels we set are [16, 32, 64, 128, 256].
"""


class ConvReLU(nn.Module):
    def __init__(self, in_data, out_data):
        super(ConvReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_data, out_data, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_data),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_data, out_data, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_data),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_data, out_data):
        super(UpConv, self).__init__()
        self.tconv = nn.Sequential(
            nn.ConvTranspose3d(in_data, out_data, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1),
            nn.BatchNorm3d(out_data),
            nn.ReLU(inplace=True),
        )
        self.conv = ConvReLU(out_data*2, out_data)

    def forward(self, x, xx):
        x = self.tconv(x)
        x = torch.cat([x, xx], dim=1)
        x = self.conv(x)
        return x


class Liangliang_Liu(nn.Module):
    def __init__(self, in_data, out_data):
        super(Liangliang_Liu, self).__init__()
        channels = [16, 32, 64, 128, 256]
        self.conv1 = ConvReLU(in_data, channels[0])
        self.conv2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            ConvReLU(channels[0], channels[1])
        )
        self.conv3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            ConvReLU(channels[1], channels[2])
        )
        self.conv4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            ConvReLU(channels[2], channels[3])
        )
        self.conv5 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            ConvReLU(channels[3], channels[4])
        )
        self.tconv1 = UpConv(channels[4], channels[3])
        self.tconv2 = UpConv(channels[3], channels[2])
        self.tconv3 = UpConv(channels[2], channels[1])
        self.tconv4 = UpConv(channels[1], channels[0])

        self.conv = nn.Sequential(
            nn.Conv3d(channels[0], out_data, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(out_data),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x = self.conv5(x4)
        x = self.tconv1(x, x4)
        x = self.tconv2(x, x3)
        x = self.tconv3(x, x2)
        x = self.tconv4(x, x1)
        x = self.conv(x)
        # print('x.shape: ', x.shape)
        return x
