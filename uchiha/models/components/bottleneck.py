import math

from torch import nn

from ..builder import BOTTLENECK


@BOTTLENECK.register_module()
class ConvBottle(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBottle, self).__init__()
        self.deconv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out


@BOTTLENECK.register_module()
class LinearBottle(nn.Module):
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
