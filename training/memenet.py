###################################################################################################
# MemeNet network
# Marco Giordano
# Center for Project Based Learning
# 2022 - ETH Zurich
###################################################################################################
"""
MemeNet network description
"""
from signal import pause
from torch import nn

import ai8x

import matplotlib
import matplotlib.pyplot as plt

"""
Network description class
"""
class MemeNet(nn.Module):
    """
    7-Layer CNN - Lightweight image classification
    """
    def __init__(self, num_classes=0, dimensions=(28, 28), num_channels=1, bias=False, **kwargs):
        super().__init__()

        # assert dimensions[0] == dimensions[1]  # Only square supported

        # Keep track of image dimensions so one constructor works for all image sizes
        dim_x, dim_y = dimensions

        self.conv1 = ai8x.FusedConv2dReLU(in_channels = num_channels, out_channels = 4, kernel_size = 3,
                                          padding=1, bias=bias, **kwargs)
        #dim=32*32*4
        
        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(in_channels = 4, out_channels = 8, kernel_size = 3,
                                          padding=1, pool_size=2,pool_stride=2,pool_dilation=1, bias=bias, **kwargs)
        dim_x //= 2 
        dim_y //= 2
        #dim=16*16*8
        
        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(in_channels = 8, out_channels = 16, kernel_size = 3,
                                          padding=1, pool_size=2,pool_stride=2,pool_dilation=1,bias=bias, **kwargs)
        dim_x //= 2  
        dim_y //= 2
        #dim=8*8*16


        self.fc1 = ai8x.FusedLinearReLU(dim_x*dim_y*16, 100, bias=bias, **kwargs)
        self.z = ai8x.FusedLinearReLU(100, dim_x*dim_y*16, bias=bias, **kwargs)
        
        '''
        ConvTranspose2d:
        Kernel sizes must be 3×3.
        Padding can be 0, 1, or 2.
        Stride is fixed to [2, 2]. Output padding is fixed to 1.
        '''
        #dim=8*8*16
        self.deconv1 = ai8x.FusedConvTranspose2dReLU(in_channels = 16, out_channels = 8, kernel_size = 3, stride=2,
                                          padding=1, bias=bias, **kwargs)
        
        dim_x *= 2  
        dim_y *= 2
        #dim=16*16*8

        self.deconv2 = ai8x.FusedConvTranspose2dReLU(in_channels = 8, out_channels = 4, kernel_size = 3, stride=2,
                                          padding=1, bias=bias, **kwargs)
        dim_x *= 2  
        dim_y *= 2
        #dim=32*32*4
        
        self.conv4 = ai8x.Conv2d(in_channels = 4, out_channels = num_channels, kernel_size = 3,
                                          padding=1, bias=bias, **kwargs)
        
        
        #dim=32x32xnum_channels
        assert dim_x == 32
        assert dim_y == 32
        assert bias == False
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    """
    Assemble the model
    """
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        # # Data plotting - for debug
        # matplotlib.use('MacOSX')
        # plt.imshow(x[1, 0], cmap="gray")
        # plt.show()
        # breakpoint()
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        #flatten
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.z(x)
        
        #unflatten
        x = x.view(x.size(0), 16, 8, 8)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.conv4(x)
        
        return x


def memenet(pretrained=False, **kwargs):
    """
    Constructs a MemeNet model.
    """
    assert not pretrained
    return MemeNet(**kwargs)

"""
Network description
"""
models = [
    {
        'name': 'memenet',
        'min_input': 1,
        'dim': 2,
    }
]

