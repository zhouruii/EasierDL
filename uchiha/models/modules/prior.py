import torch
import torch.nn as nn

from .common import conv3x3
from ..builder import MODULE


@MODULE.register_module()
class GRCPBranch(nn.Module):
    def __init__(self, in_channels, num_heads=8, strategy='dilation'):
        super(GRCPBranch, self).__init__()
        self.in_channels = in_channels
        self.strategy = strategy

        self.pre_dwconv = conv3x3(in_channels, 1)

        if strategy == 'dilation':
            self.level1 = conv3x3(3, 1, 1, 1, 1)
            self.level2 = conv3x3(3, 1, 1, 1, 2)
            self.level3 = conv3x3(3, 1, 1, 1, 3)

        self.fusion_conv = conv3x3(3, num_heads)
        self.act = nn.ReLU()
        self.post_conv = conv3x3(num_heads, in_channels)

    def forward(self, x):
        # x.shape (B, C, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = self.pre_dwconv(x)
        x = torch.cat([avg_out, max_out, x], dim=1)

        stage1 = self.level1(x)
        stage2 = self.level2(x)
        stage3 = self.level3(x)
        multi_scale_feat = torch.cat([stage1, stage2, stage3], dim=1)

        attn = self.fusion_conv(multi_scale_feat)
        attn = self.act(attn)

        mask = attn.unsqueeze(2)  # B nH 1 H W
        out = self.post_conv(attn)

        return out, mask
