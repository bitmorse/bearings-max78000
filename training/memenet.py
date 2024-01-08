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
    def __init__(self, num_classes=10, dimensions=(28, 28), num_channels=1, bias=False, **kwargs):
        super().__init__()

        # assert dimensions[0] == dimensions[1]  # Only square supported

        # Keep track of image dimensions so one constructor works for all image sizes
        dim_x, dim_y = dimensions

        self.conv1 = ai8x.FusedConv2dReLU(in_channels = num_channels, out_channels = 32, kernel_size = 3,
                                          padding=1, bias=bias, **kwargs)
        # padding 1 -> no change in dimensions

        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(in_channels = 32, out_channels = 24, kernel_size = 3,
                                          padding=1, bias=bias, **kwargs)
        dim_x //= 2  # pooling, padding 0
        dim_y //= 2
        # conv padding 1 -> no change in dimensions

        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(in_channels = 24, out_channels = 24, kernel_size = 3,
                                          padding=1, bias=bias, **kwargs)
        dim_x //= 2  # pooling, padding 0
        dim_y //= 2

        self.conv4 = ai8x.FusedMaxPoolConv2dReLU(in_channels = 24, out_channels = 24, kernel_size = 3,
                                          padding=1, bias=bias, **kwargs)
        dim_x //= 2  # pooling, padding 0
        dim_y //= 2

        # conv padding 1 -> no change in dimensions
        self.conv5 = ai8x.FusedConv2dReLU(in_channels = 24, out_channels = 24, kernel_size = 3,
                                          padding=1, bias=bias, **kwargs)
        # padding 1 -> no change in dimensions

        self.mp = ai8x.MaxPool2d(kernel_size = 2, **kwargs)
        dim_x //= 2  # pooling, padding 0
        dim_y //= 2

        self.fc1 = ai8x.FusedLinearReLU(dim_x*dim_y*24, 32, bias=True, **kwargs)

        self.fc2 = ai8x.FusedLinearReLU(32, 16, bias=True, **kwargs)

        self.fc3 = ai8x.Linear(16, num_classes, wide=True, bias=True, **kwargs)

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
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.mp(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        # Loss chosed, CrossEntropyLoss, takes softmax into account already func.log_softmax(x, dim=1))

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

