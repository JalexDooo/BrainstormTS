import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
        正卷积
    """
    def __init__(self, input_data, output_data):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(input_data, output_data, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(output_data),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(output_data, output_data, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(output_data),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class ConvTransBlock(nn.Module):
    def __init__(self, input_data, output_data):
        super(ConvTransBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose3d(input_data, output_data, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1),
            nn.BatchNorm3d(output_data),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        x = self.conv1(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, input_data, output_data):
        super(UpBlock, self).__init__()
        self.up = ConvTransBlock(input_data, output_data)
        self.down = ConvBlock(2*output_data, output_data)
    
    def forward(self, x, down_features):
        x = self.up(x)
        x = torch.cat([x, down_features], dim=1) # 横向拼接
        x = self.down(x)
        return x

def maxpool():
    pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
    return pool

class SingleConvBlock(nn.Module):
    """
        正卷积
    """
    def __init__(self, input_data, output_data):
        super(SingleConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(input_data, output_data, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(output_data),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class SingleTransConvBlock(nn.Module):
    def __init__(self, input_data, output_data):
        super(SingleTransConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose3d(input_data, output_data, kernel_size=3, stride=2, padding=1, output_padding=1,
                               dilation=1),
            nn.BatchNorm3d(output_data),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvBlockWithKernel3(nn.Module):
    def __init__(self, input_data, output_data):
        super(ConvBlockWithKernel3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(input_data, output_data, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(output_data),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class ConvBlockWithKernel5(nn.Module):
    def __init__(self, input_data, output_data):
        super(ConvBlockWithKernel5, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(input_data, output_data, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm3d(output_data),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class ConvBlockWithKernel7(nn.Module):
    def __init__(self, input_data, output_data):
        super(ConvBlockWithKernel7, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(input_data, output_data, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm3d(output_data),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class DoubleScaleUnit(nn.Module):
    def __init__(self, input_data, output_data):
        super(DoubleScaleUnit, self).__init__()
        self.weight1 = nn.Parameter(torch.ones(1))
        self.weight2 = nn.Parameter(torch.ones(1))

        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockWithKernel3(input_data, output_data))
        self.conv.append(ConvBlockWithKernel5(input_data, output_data))



    def forward(self, x):
        x = self.weight1*self.conv[0](x) + self.weight2*self.conv[1](x)

        return x

