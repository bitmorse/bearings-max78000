###################################################################################################
# BearingNet network
# Sam Sulaimanov
# 2023
###################################################################################################
"""
BearingNet network description
"""
from signal import pause
from torch import nn

import ai8x


"""
Network description class
"""
class BearingNet(nn.Module):
    """
    7-Layer CNN - Lightweight image classification
    """
    def __init__(self, num_classes=0, dimensions=(64, 64), num_channels=1, bias=False, **kwargs):
        super().__init__()

        assert dimensions[0] == dimensions[1]  # Only square supported
        assert dimensions[0] in [64]  # Only these sizes supported
        
        #print("num_channels: ",num_channels)
        #print(dimensions)

        # Keep track of image dimensions so one constructor works for all image sizes
        dim_x, dim_y = dimensions

        self.conv1 = ai8x.FusedConv2dReLU(in_channels = num_channels, out_channels = 64, kernel_size = 3,
                                          padding=1, bias=bias, **kwargs)
        #dim=64*64*64
        
        
        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(in_channels = 64, out_channels = 32, kernel_size = 3,
                                          padding=1, pool_size=2,pool_stride=2,pool_dilation=1, bias=bias, **kwargs)
        dim_x //= 2 
        dim_y //= 2
        #dim=32*32*4
        
        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(in_channels = 32, out_channels = 16, kernel_size = 3,
                                          padding=1, pool_size=2,pool_stride=2,pool_dilation=1, bias=bias, **kwargs)
        dim_x //= 2 
        dim_y //= 2
        #dim=16*16*8
        
        self.latent_unflatten_dim = 12
        
        self.conv4 = ai8x.FusedMaxPoolConv2dReLU(in_channels = 16, out_channels = self.latent_unflatten_dim, kernel_size = 3,
                                          padding=1, pool_size=2,pool_stride=2,pool_dilation=1,bias=bias, **kwargs)
        dim_x //= 2  
        dim_y //= 2
        #dim=8*8*2


        self.fc1 = ai8x.FusedLinearReLU(dim_x*dim_y*self.latent_unflatten_dim, 2, bias=bias, **kwargs)
        self.z = ai8x.FusedLinearReLU(2, dim_x*dim_y*self.latent_unflatten_dim, bias=bias, **kwargs)
        
        '''
        ConvTranspose2d:
        Kernel sizes must be 3x3.
        Padding can be 0, 1, or 2.
        Stride is fixed to [2, 2]. Output padding is fixed to 1.
        '''
        #dim=8*8*2
        self.deconv1 = ai8x.FusedConvTranspose2dReLU(in_channels = self.latent_unflatten_dim, out_channels = 16, kernel_size = 3, stride=2,
                                          padding=1, bias=bias, **kwargs)
        
        dim_x *= 2  
        dim_y *= 2
        #dim=16*16*8

        self.deconv2 = ai8x.FusedConvTranspose2dReLU(in_channels = 16, out_channels = 32, kernel_size = 3, stride=2,
                                          padding=1, bias=bias, **kwargs)
        dim_x *= 2  
        dim_y *= 2
        #dim=32*32*4
        
        self.deconv3 = ai8x.FusedConvTranspose2dReLU(in_channels = 32, out_channels = 64, kernel_size = 3, stride=2,
                                          padding=1, bias=bias, **kwargs)
        dim_x *= 2  
        dim_y *= 2
        #dim=64*64*4
        
        self.conv5 = ai8x.Conv2d(in_channels = 64, out_channels = num_channels, kernel_size = 3,
                                          padding=1, bias=bias, wide=True, **kwargs)
        
        
        assert dim_x == 64
        assert dim_y == 64
        assert bias == False
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    """
    Assemble the model
    """
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        #flatten
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.z(x)
        
        #unflatten to dim=8*8*4
        x = x.view(x.size(0), self.latent_unflatten_dim, 8, 8)

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.conv5(x)
        
        return x


def bearingnet(pretrained=False, **kwargs):
    """
    Constructs a BearingNet model.
    """
    assert not pretrained
    return BearingNet(**kwargs)

"""
Network description
"""
models = [
    {
        'name': 'bearingnet',
        'min_input': 1,
        'dim': 2,
    }
]

