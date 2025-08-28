import torch
import torch.nn as nn
import torch.nn.functional as F

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


class PixelAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        hidden_channels = in_channels // 2
        self.pixel_attn = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pixel_attn(x)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, strategy='dilation'):
        super().__init__()
        self.in_channels = in_channels
        self.strategy = strategy

        # ---------------------- Pooling ---------------------- #
        self.conv_pool = conv3x3(in_channels, 1)

        # ---------------------- Multi-scale ---------------------- #
        if strategy == 'dilation':
            self.level1 = conv3x3(3, 1, 1, 1, 1)
            self.level2 = conv3x3(3, 1, 1, 1, 2)
            self.level3 = conv3x3(3, 1, 1, 1, 3)

        self.fusion_conv = conv3x3(3, 1)
        self.act = nn.Sigmoid()
        self.post_conv = conv3x3(in_channels, in_channels)

    def forward(self, x):
        # x.shape (B, C, H, W)
        res = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = self.conv_pool(x)
        x = torch.cat([avg_out, max_out, x], dim=1)  # B 3 H W

        stage1 = self.level1(x)
        stage2 = self.level2(x)
        stage3 = self.level3(x)
        multi_scale_attn = torch.cat([stage1, stage2, stage3], dim=1)

        attn = self.fusion_conv(multi_scale_attn)
        attn = self.act(attn)

        return res * attn


@MODULE.register_module()
class MultiGRCPBranch(nn.Module):
    def __init__(self, in_channels, strategy='dilation', ds_scale=1):
        super().__init__()
        self.in_channels = in_channels
        self.strategy = strategy
        self.ds_scale = ds_scale

        self.gp_branch = nn.Sequential(PixelAttention(in_channels=in_channels),
                                       SpatialAttention(in_channels=in_channels, strategy=strategy))
        self.rp_branch = nn.Sequential(PixelAttention(in_channels=in_channels),
                                       SpatialAttention(in_channels=in_channels, strategy=strategy))
        self.hp_branch = nn.Sequential(PixelAttention(in_channels=in_channels),
                                       SpatialAttention(in_channels=in_channels, strategy=strategy))

        # --------------------------------- Mask(Downsample) --------------------------------- #
        self.gp_conv = conv3x3(in_channels, 1)
        self.rp_conv_ll = conv3x3(in_channels, 1)
        self.rp_conv_lh = conv3x3(in_channels, 1)
        self.rp_conv_hl = conv3x3(in_channels, 1)
        self.hp_conv = conv3x3(in_channels, 1)

        self.gp_post_conv = conv3x3(in_channels, in_channels)
        self.rp_post_conv = conv3x3(in_channels, in_channels)
        self.hp_post_conv = conv3x3(in_channels, in_channels)

    def forward(self, x):
        gp, rp, hp = x
        gp = self.gp_branch(gp)
        rp = self.rp_branch(rp)
        hp = self.hp_branch(hp)

        # --------------------------------- Mask(Downsample) --------------------------------- #
        gp_mask = self.gp_conv(gp)
        rp_mask_ll = self.rp_conv_ll(rp)
        rp_mask_lh = self.rp_conv_lh(rp)
        rp_mask_hl = self.rp_conv_hl(rp)
        hp_mask = self.hp_conv(hp)
        gp_mask = F.interpolate(gp_mask, scale_factor=pow(0.5, self.ds_scale - 1), mode='bicubic')
        rp_mask_ll = F.interpolate(rp_mask_ll, scale_factor=pow(0.5, self.ds_scale), mode='bicubic')
        rp_mask_lh = F.interpolate(rp_mask_lh, scale_factor=pow(0.5, self.ds_scale), mode='bilinear')
        rp_mask_hl = F.interpolate(rp_mask_hl, scale_factor=pow(0.5, self.ds_scale), mode='bilinear')
        hp_mask = F.interpolate(hp_mask, scale_factor=pow(0.5, self.ds_scale), mode='bicubic')

        gp = self.gp_post_conv(gp)
        rp = self.rp_post_conv(rp)
        hp = self.hp_post_conv(hp)

        return (gp, rp, hp), (gp_mask, rp_mask_ll, rp_mask_lh, rp_mask_hl, hp_mask)
