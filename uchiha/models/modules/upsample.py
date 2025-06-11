import math

from torch import nn

from uchiha.models.builder import MODULE


@MODULE.register_module()
class PixelShuffle(nn.Module):
    """ `PixelShuffle` to image and sequence

    After `PixelShuffle`, a Conv/Linear layer id used to map the data to the output dimension.

    Args:
        factor (int): upsample factor
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
    """

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


@MODULE.register_module()
class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_channels=128, factor=2):
        super(PixelShuffleUpsample, self).__init__()
        self.in_channel = in_channels

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * factor, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(factor))

    def forward(self, x):
        return self.body(x)
