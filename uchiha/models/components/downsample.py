import math

from torch import nn

from ..builder import DOWNSAMPLE


@DOWNSAMPLE.register_module()
class PixelUnShuffle(nn.Module):

    def __init__(self, factor=2, in_channel=512, out_channel=1024):
        super().__init__()
        self.downsample = nn.PixelUnshuffle(downscale_factor=factor)
        self.fc = nn.Linear(in_channel * 4, out_channel)

    def forward(self, x):
        B, L, C = x.shape
        H, W = int(math.sqrt(L)), int(math.sqrt(L))
        downsample = self.downsample(x.view(B, C, H, W))
        fc = self.fc(downsample.flatten(2).transpose(1, 2))
        return fc
