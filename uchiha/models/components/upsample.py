import math

from torch import nn

from ..builder import UPSAMPLE


@UPSAMPLE.register_module()
class PixelShuffle(nn.Module):

    def __init__(self, factor=2, in_channel=1024, out_channel=512):
        super().__init__()
        self.upsample = nn.PixelShuffle(upscale_factor=factor)
        self.fc = nn.Linear(in_channel // 4, out_channel)

    def forward(self, x):
        B, L, C = x.shape
        H, W = int(math.sqrt(L)), int(math.sqrt(L))
        upsample = self.upsample(x.view(B, C, H, W))
        fc = self.fc(upsample.flatten(2).transpose(1, 2))
        return fc
