import torch
import torch.nn as nn
import math

from .BasicModule import *

class MultiscaleUNet3D(nn.Module):
    def __init__(self, input_data=4, output_data=2, degree=16):
        super(MultiscaleUNet3D, self).__init__()

        # kn = [] # [degree, degree*2, degree*4, degree*8, ...] [16, 32, 64]
        # for i in range(4):
        #     kn.append((2 ** i) * degree)
        kn = [32, 128, 256, 384]
        # kn = [1, 1, 1, 1]

        self.pre_layer = SingleConvBlock(input_data, kn[0])
        self.unit1 = nn.Sequential(
            DoubleScaleUnit(kn[0], kn[1]),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        )
        self.unit2 = nn.Sequential(
            DoubleScaleUnit(kn[1], kn[2]),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        )
        self.mid_layer = SingleConvBlock(kn[2], kn[3])
        self.up_unit1 = UpBlock(kn[3], kn[2])
        self.up_unit2 = UpBlock(kn[2], kn[1])
        self.up_unit3 = UpBlock(kn[1], kn[0])

        self.out_layer = nn.Sequential(
            SingleTransConvBlock(kn[0], kn[0] * 2),
            nn.Conv3d(kn[0] * 2, output_data, kernel_size=3, stride=1, padding=1)
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

        # in:torch.Size([batch, 4, 144, 192, 192])
        x0 = self.pre_layer(x)
        # print('pre_layer.shape : ', x0.shape)

        x1 = self.unit1(x0)
        # print('unit1.shape : ', x1.shape)

        x2 = self.unit2(x1)
        # print('unit2.shape : ', x2.shape)

        mid = self.mid_layer(x2)
        # print('mid_layer.shape : ', mid.shape)

        x = self.up_unit1(mid, x2)
        # print('up_unit1.shape : ', x.shape)

        x = self.up_unit2(x, x1)
        # print('up_unit2.shape : ', x.shape)

        x = self.up_unit3(x, x0)
        # print('un_unit3.shape : ', x.shape)

        out = self.out_layer(x)
        # print('out_layer.shape : ', out.shape)
        """
        pre_layer.shape :  torch.Size([1, 32, 72, 96, 96])
        unit1.shape :  torch.Size([1, 128, 36, 48, 48])
        unit2.shape :  torch.Size([1, 256, 18, 24, 24])
        mid_layer.shape :  torch.Size([1, 416, 9, 12, 12])
        up_unit1.shape :  torch.Size([1, 256, 18, 24, 24])
        up_unit2.shape :  torch.Size([1, 128, 36, 48, 48])
        un_unit3.shape :  torch.Size([1, 32, 72, 96, 96])
        pre_out_layer.shape :  torch.Size([1, 64, 144, 192, 192])
        out_layer.shape :  torch.Size([1, 2, 144, 192, 192])
        """

        return out


'''
----------------------epoch 56--------------------
train_loss : 0.006298362467624602
train_dice : 0.8203808439125962
----------------------epoch 57--------------------
train_loss : 0.006016374226836281
train_dice : 0.8454236551921788
----------------------epoch 58--------------------
train_loss : 0.006197508394098436
train_dice : 0.850454255872331
----------------------epoch 59--------------------
train_loss : 0.005685358240287185
train_dice : 0.8277880780209701
'''