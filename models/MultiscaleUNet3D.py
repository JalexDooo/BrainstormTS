import torch
import torch.nn as nn
import math

from .BasicModule import *

class MultiscaleUNet3D(nn.Module):

    def __init__(self, input_data=4, output_data=2, degree=8):
        super(MultiscaleUNet3D, self).__init__()

        self.pre_layer1 = SingleConvBlock(input_data, degree)


    def forward(self, x):
        x = self.pre_layer1(x)

        return x



