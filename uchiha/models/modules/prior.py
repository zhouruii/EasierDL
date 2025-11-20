import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward

from .common import conv3x3, build_act
from .basic_resnet import BasicResidualBlock
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
    def __init__(self, in_channels, act=None):
        super().__init__()
        self.in_channels = in_channels

        hidden_channels = in_channels // 2
        self.pixel_attn = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid() if act is None else act()
        )

    def forward(self, x):
        attn = self.pixel_attn(x)
        return x * attn + x


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, strategy='dilation', act=None):
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
        self.act = nn.Sigmoid() if act is None else act()

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

        return res * attn + res  # res.mul_(attn) +res


@MODULE.register_module()
class MultiGRCPBranch(nn.Module):
    def __init__(self, in_channels, strategy='dilation', act=None, ds_scale=1):
        super().__init__()
        self.in_channels = in_channels
        self.strategy = strategy
        self.ds_scale = ds_scale
        self.act = build_act(act)

        self.gp_branch = self.get_attn_opt(use_rb=True, use_pixel_att=False, use_spatial_att=False)
        self.rp_branch = self.get_attn_opt(use_rb=True, use_pixel_att=False, use_spatial_att=False)
        self.hp_branch = self.get_attn_opt(use_rb=True, use_pixel_att=False, use_spatial_att=False)

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

    def get_attn_opt(self, use_rb=True, use_pixel_att=True, use_spatial_att=True):
        if use_rb:
            # 使用残差块进行先验的特征提取
            return BasicResidualBlock(self.in_channels, self.in_channels)

        if not use_pixel_att and not use_spatial_att:
            return nn.Identity()

        attn_opts = []
        if use_pixel_att:
            attn_opts.append(PixelAttention(self.in_channels, self.act))
        if use_spatial_att:
            attn_opts.append(SpatialAttention(self.in_channels, self.strategy, self.act))

        return nn.Sequential(*attn_opts)


@MODULE.register_module()
class RainHazeGRCPBranch(nn.Module):
    def __init__(self, in_channels, num_heads, strategy='dilation', act=None, ds_scale=1, last_prior=False):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.strategy = strategy
        self.ds_scale = ds_scale
        self.act = build_act(act)
        self.last_prior = last_prior

        # BasicResidualBlock(self.in_channels, self.in_channels)
        # PixelAttention(self.in_channels, self.act)
        # SpatialAttention(self.in_channels, self.strategy, self.act)
        self.rain_branch = SpatialAttention(self.in_channels, self.strategy, self.act)
        self.haze_branch = SpatialAttention(self.in_channels, self.strategy, self.act)

        # --------------------------------- Mask(Downsample) --------------------------------- #
        self.rp_conv_ll = conv3x3(in_channels, self.num_heads)
        self.act_rp = nn.Sigmoid()
        self.hp_conv = conv3x3(in_channels, 1)
        self.act_hp = nn.Sigmoid()

    def forward(self, x):
        rp, hp = x
        rp = self.rain_branch(rp)
        hp = self.haze_branch(hp)

        # --------------------------------- Mask(Downsample) --------------------------------- #
        rp_mask = self.act_rp(self.rp_conv_ll(rp))
        hp_mask = self.act_hp(self.hp_conv(hp))
        rp_mask = F.interpolate(rp_mask, scale_factor=pow(0.5, self.ds_scale-1), mode='bicubic')
        hp_mask = F.interpolate(hp_mask, scale_factor=pow(0.5, self.ds_scale), mode='bicubic')

        if self.last_prior:
            return None, (rp_mask.unsqueeze(2), hp_mask)
        else:
            return (rp, hp), (rp_mask.unsqueeze(2), hp_mask)
