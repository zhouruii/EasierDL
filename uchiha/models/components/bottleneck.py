import math

from torch import nn

from ..builder import BOTTLENECK


@BOTTLENECK.register_module()
class ConvBottle(nn.Module):
    """ bottleneck based on `Conv` between encoder and decoder

    Args:
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
        kernel_size (int): Kernel size of `Conv`
        stride (int): Stride of `Conv`
        padding (int): Padding size of `Conv`
    """

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super(ConvBottle, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.GELU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.GELU(),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out


@BOTTLENECK.register_module()
class LinearBottle(nn.Module):
    """ bottleneck based on `Linear` between encoder and decoder

    Args:
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
    """

    def __init__(self, in_channel, out_channel):
        super(LinearBottle, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.GELU(),
            nn.Linear(out_channel, out_channel),
            nn.GELU(),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        return self.fc(x)
