import math

import torch
from timm.layers import to_2tuple
from torch import nn

from uchiha.models.builder import MODULE


@MODULE.register_module()
class PixelUnShuffle(nn.Module):
    """ `PixelUnShuffle` to image and sequence

    After `PixelUnShuffle`, a `Conv`/`Linear` layer is used to map the data to the output dimension.

    Args:
        factor (int): downsample factor
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
    """

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


@MODULE.register_module()
class DownsampleConv(nn.Module):
    """ `Conv` with stride > 1 to downsample image

    Args:
        in_channel (int): Number of input channels. Default: 512
        out_channel (int): Number of output channels. Default: 1024
        kernel_size (int): Kernel size for `Conv`. Default: 3
        stride (int): Stride for `Conv`. Default: 2
        padding (int): Padding for `Conv`. Default: 1
    """

    def __init__(self,
                 in_channel=512,
                 out_channel=1024,
                 kernel_size=3,
                 stride=2,
                 padding=1):

        super().__init__()
        self.downsample = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
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


@MODULE.register_module()
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        in_channel (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, in_channel, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = to_2tuple(input_resolution)
        self.dim = in_channel
        self.reduction = nn.Linear(4 * in_channel, 2 * in_channel, bias=False)
        self.norm = norm_layer(4 * in_channel)

    def forward(self, x):
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

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


@MODULE.register_module()
class PixelShuffleDownsample(nn.Module):
    def __init__(self, in_channels=128, factor=2):
        super(PixelShuffleDownsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(in_channels, in_channels // factor, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(factor))

    def forward(self, x):
        return self.body(x)
