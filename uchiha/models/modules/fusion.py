import torch
from torch import nn

from uchiha.models.builder import MODULE


@MODULE.register_module()
class CatPWConvFusion(nn.Module):
    """ Concat followed by PW-Conv

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels

        if out_channels is None:
            out_channels = in_channels // 2
        self.fusion = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, y):
        # x y.shape: (B, C, H, W )
        cat = torch.cat([x, y], dim=1)
        out = self.fusion(cat)  # B H*W C

        return out

