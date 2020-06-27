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
    def __init__(self, input_data, output_data, stride=2):
        super(SingleTransConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose3d(input_data, output_data, kernel_size=3, stride=stride, padding=1, output_padding=1,
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
            nn.RReLU(lower=0.125, upper=1.0/3, inplace=True)
            # nn.ReLU(inplace=True)
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

class DilationConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DilationConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, dilation=2, padding=2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class TransConvBlock(nn.Module):
    def __init__(self, input_data, output_data, stride=2):
        super(TransConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose3d(input_data, output_data, kernel_size=3, stride=stride, padding=1, output_padding=stride-1,
                               dilation=1),
            nn.BatchNorm3d(output_data),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    
class FullResBlock(nn.Module):
    def __init__(self, input_data, output_data, stride=2):
        super(FullResBlock, self).__init__()
        self.resblock = nn.Sequential(
            nn.Conv3d(input_data, output_data, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(output_data),
            nn.ReLU(inplace=True),
            nn.Conv3d(output_data, output_data, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(output_data)
        )
        self.conv = nn.Sequential(
            nn.Conv3d(input_data, output_data, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(output_data)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.resblock(x)
        x = self.conv(x)
        x += res
        x = self.relu(x)
        return x

class NewResBlock(nn.Module):
    def __init__(self, input_data, output_data):
        super(NewResBlock, self).__init__()
        self.resblock = nn.Sequential(
            nn.BatchNorm3d(input_data),
            nn.RReLU(lower=0.125, upper=1.0/3, inplace=True),
            nn.Conv3d(input_data, output_data, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(output_data),
            nn.RReLU(lower=0.125, upper=1.0 / 3, inplace=True),
            nn.Conv3d(output_data, output_data, kernel_size=3, stride=1, padding=1)
        )
        self.conv = nn.Sequential(
            nn.BatchNorm3d(input_data),
            nn.RReLU(lower=0.125, upper=1.0 / 3, inplace=True),
            nn.Conv3d(input_data, output_data, kernel_size=3, stride=1, padding=1)
        )
        
    def forward(self, x):
        res = self.resblock(x)
        x = self.conv(x)
        x += res
        return x

class NewConvDown(nn.Module):
    def __init__(self, input_data, output_data, stride=2):
        super(NewConvDown, self).__init__()
        self.conv = nn.Conv3d(input_data, output_data, kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class NewConvUp(nn.Module):
    def __init__(self, input_data, output_data, stride=2):
        super(NewConvUp, self).__init__()
        self.conv = nn.ConvTranspose3d(input_data, output_data, kernel_size=3, stride=stride, padding=1, output_padding=1, dilation=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, input_data, output_data):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm3d(input_data),
            nn.RReLU(lower=0.125, upper=1.0 / 3, inplace=True),
            nn.Conv3d(input_data, output_data, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
