import math

from einops import rearrange
from torch import nn
import torch.nn.functional as F

from uchiha.models.modules.cbam import BasicECA1d
from ..builder import MODULE


@MODULE.register_module()
class GDFN(nn.Module):
    """
    GDFN: (Gated-Dconv Feed-Forward Network), refer to
    Restormer: Efficient Transformer for High-Resolution Image Restoration https://arxiv.org/abs/2111.09881
    """

    def __init__(self, in_channels, ffn_expansion_factor, bias):
        super(GDFN, self).__init__()

        hidden_features = int(in_channels * ffn_expansion_factor)

        self.project_in = nn.Conv2d(in_channels, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, in_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


@MODULE.register_module()
class LeFF(nn.Module):
    """
    LeFF: Locally-enhanced  Feed-Forward Network, refer to
    Uformer: A General U-Shaped Transformer for Image Restoration
    """

    def __init__(self, in_channels=32, ratio=4, act_layer=nn.GELU, use_eca=False):
        super().__init__()
        hidden_dim = in_channels * ratio

        self.linear1 = nn.Sequential(nn.Linear(in_channels, hidden_dim),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, in_channels))
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.eca = BasicECA1d(in_channels) if use_eca else nn.Identity()

    def forward(self, x):
        # bs x hw x c
        B, C, H, W = x.size()
        x = rearrange(x, 'b c h w -> b (h w) c', h=H, w=W)

        x = self.linear1(x)
        # spatial restore
        x = rearrange(x, 'b (h w) (c) -> b c h w ', h=H, w=W)
        x = self.dwconv(x)
        # flatten
        x = rearrange(x, ' b c h w -> b (h w) c', h=H, w=W)
        x = self.linear2(x)
        x = self.eca(x)

        x = rearrange(x, 'b (h w) c -> b c h w ', h=H, w=W)

        return x

    def flops(self, H, W):
        flops = 0
        # fc1
        flops += H * W * self.in_channels * self.hidden_dim
        # dwconv
        flops += H * W * self.hidden_dim * 3 * 3
        # fc2
        flops += H * W * self.hidden_dim * self.in_channels
        print("LeFF:{%.2f}" % (flops / 1e9))
        # eca
        if hasattr(self.eca, 'flops'):
            flops += self.eca.flops()
        return flops


@MODULE.register_module()
class PriorGuidedFFN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
