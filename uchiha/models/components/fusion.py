import math

from torch import nn

from ..builder import FUSION


@FUSION.register_module()
class CatConv(nn.Module):
    """ Concat followed by Conv

    Args:
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
    """

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):

        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.GELU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.GELU()
        )

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.conv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C

        return x


@FUSION.register_module()
class CatConvs(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fusions = nn.ModuleList()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_fusions = len(self.in_channels)
        for i in range(self.num_fusions):
            self.fusions.append(CatConv(self.in_channels[i], self.out_channels[i]))


@FUSION.register_module()
class CatLinear(nn.Module):
    """ Concat followed by Linear

    Args:
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
    """
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channel, out_channel, bias=False),
            nn.GELU(),
            nn.Linear(out_channel, out_channel, bias=False),
            nn.GELU()
        )

    def forward(self, x):
        return self.fc(x)


@FUSION.register_module()
class CatLinears(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fusions = nn.ModuleList()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_fusions = len(self.in_channels)
        for i in range(self.num_fusions):
            self.fusions.append(CatLinear(self.in_channels[i], self.out_channels[i]))
