import math

import torch
from torch import nn

from ..builder import DOWNSAMPLE


@DOWNSAMPLE.register_module()
class PixelUnShuffle(nn.Module):

    def __init__(self, factor=2, in_channel=512, out_channel=1024):
        super().__init__()
        self.downsample = nn.PixelUnshuffle(downscale_factor=factor)
        self.fc = nn.Linear(in_channel * 4, out_channel)
        self.norm = nn.LayerNorm(out_channel)

    def forward(self, x):
        B, L, C = x.shape
        H, W = int(math.sqrt(L)), int(math.sqrt(L))
        downsample = self.downsample(x.view(B, C, H, W))
        fc = self.fc(downsample.flatten(2).transpose(1, 2))
        return self.norm(fc)


@DOWNSAMPLE.register_module()
class Stride2Conv(nn.Module):
    def __init__(self, in_channel=512, out_channel=1024):
        super().__init__()
        self.downsample = nn.Conv2d(in_channel, out_channel, 3, 2, 1)
        self.activate = nn.GELU()

    def forward(self, x):
        if len(x.shape) == 3:
            B, L, C = x.shape
            H, W = int(math.sqrt(L)), int(math.sqrt(L))
            x = x.view(B, H, W, C).permute(0, 3, 1, 2)
            out = self.activate(self.downsample(x))
            return out.flatten(2).transpose(1, 2)
        else:
            return self.activate(self.downsample(x))


@DOWNSAMPLE.register_module()
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x
