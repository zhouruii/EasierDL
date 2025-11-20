"""
Single image de-raining by multi-scale Fourier Transform network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import MODEL


def conv(in_channels, out_channels, kernel_size, bias=True, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


# Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=3):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


# Dual Attention Block (DAB)
class DAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True)):

        super(DAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)

        self.SA = spatial_attn_layer()  ## Spatial Attention
        self.CA = CALayer(n_feat, reduction)  ## Channel Attention
        self.body = nn.Sequential(*modules_body)
        self.conv1x1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1)

    def forward(self, x):
        res = self.body(x)
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res = torch.cat([sa_branch, ca_branch], dim=1)
        res = self.conv1x1(res)
        res += x
        return res


# Recursive Residual Group (RRG)
class RRG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, num_dab):
        super(RRG, self).__init__()
        modules_body = []
        modules_body = [DAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=act) for _ in range(num_dab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


@MODEL.register_module()
class DeRainNet(nn.Module):
    def __init__(self, inp_chans=3, n_feats=64):
        super(DeRainNet, self).__init__()
        num_dabs = 6
        num_dab = 5
        kernel_size = 3
        reduction = 16
        act = nn.PReLU(n_feats)

        Enhance1 = [conv(inp_chans, n_feats, kernel_size=kernel_size, stride=1),
                    RRG(conv, 64, 3, 16, act=nn.LeakyReLU(0.2, inplace=True), num_dab=num_dabs),
                    RRG(conv, 64, 3, 16, act=nn.LeakyReLU(0.2, inplace=True), num_dab=num_dabs),
                    conv(64, inp_chans, kernel_size=kernel_size, stride=1)]
        self.Enhance1 = nn.Sequential(*Enhance1)

        Enhance2 = [conv(inp_chans, n_feats, kernel_size=kernel_size, stride=1),
                    RRG(conv, 64, 3, 16, act=nn.LeakyReLU(0.2, inplace=True), num_dab=num_dabs),
                    RRG(conv, 64, 3, 16, act=nn.LeakyReLU(0.2, inplace=True), num_dab=num_dabs),
                    conv(64, inp_chans, kernel_size=kernel_size, stride=1)]
        self.Enhance2 = nn.Sequential(*Enhance2)

        Enhance2A = [nn.Conv2d(inp_chans, 64, 3, 1, 1),
                     RRG(conv, n_feats, kernel_size, reduction, act=act, num_dab=num_dab)]
        self.Enhance2a = nn.Sequential(*Enhance2A)

        Enhance3 = [conv(n_feats, n_feats, kernel_size=kernel_size, stride=1),
                    RRG(conv, n_feats, kernel_size, reduction, act=act, num_dab=num_dab),
                    RRG(conv, n_feats, kernel_size, reduction, act=act, num_dab=num_dab),
                    conv(n_feats, n_feats, kernel_size=kernel_size, stride=1)]
        self.Enhance3 = nn.Sequential(*Enhance3)

        inBlock1 = [conv(n_feats, n_feats, kernel_size=kernel_size, stride=1),
                    RRG(conv, n_feats, kernel_size, reduction, act=act, num_dab=num_dab),
                    RRG(conv, n_feats, kernel_size, reduction, act=act, num_dab=num_dab)]
        self.inBlock1 = nn.Sequential(*inBlock1)

        inBlock2 = [conv(n_feats * 2, n_feats, kernel_size=kernel_size, stride=1),
                    RRG(conv, n_feats, kernel_size, reduction, act=act, num_dab=num_dab),
                    RRG(conv, n_feats, kernel_size, reduction, act=act, num_dab=num_dab)]
        self.inBlock2 = nn.Sequential(*inBlock2)

        modules_tail = [RRG(conv, n_feats, kernel_size, reduction, act=act, num_dab=num_dab),
                        RRG(conv, n_feats, kernel_size, reduction, act=act, num_dab=num_dab),
                        conv(n_feats, inp_chans, kernel_size)]
        self.tail = nn.Sequential(*modules_tail)

        self.n_levels = 2
        self.scale = 0.5

    def forward(self, Rain):
        x_fft = torch.fft.fft2(Rain, dim=(-2, -1))
        # 振幅 amplitude = sqrt(real^2 + imag^2)
        Rain_amp = torch.abs(x_fft)
        # 相位 phase = atan2(imag, real)
        Rain_phase = torch.angle(x_fft)

        DeRainAMP = self.Enhance1(Rain_amp) + Rain_amp
        DeRainPHASE = self.Enhance2(Rain_phase) + Rain_phase
        clear_features1 = torch.fft.ifft2(DeRainAMP * torch.exp(1j * DeRainPHASE), dim=(-2, -1)).real * torch.fft.ifft2(
            DeRainAMP * torch.exp(1j * DeRainPHASE), dim=(-2, -1)).real
        clear_features2 = torch.fft.ifft2(DeRainAMP * torch.exp(1j * DeRainPHASE), dim=(-2, -1)).imag * torch.fft.ifft2(
            DeRainAMP * torch.exp(1j * DeRainPHASE), dim=(-2, -1)).imag
        clear_features = self.Enhance2a(torch.sqrt(clear_features1 + clear_features2))

        input_pre = None
        out = None
        for level in range(self.n_levels):

            scale = self.scale ** (self.n_levels - level - 1)
            n, c, h, w = Rain.shape
            hi = int(round(h * scale))
            wi = int(round(w * scale))
            if level == 0:
                input_clear = F.interpolate(clear_features, (hi, wi), mode='bilinear')
                first_scale_inblock = self.inBlock1(input_clear)
            else:
                input_clear = F.interpolate(clear_features, (hi, wi), mode='bilinear')
                input_pred = F.interpolate(input_pre, (hi, wi), mode='bilinear')
                inp_all = torch.cat((input_clear, input_pred), 1)
                first_scale_inblock = self.inBlock2(inp_all)

            input_pre = self.Enhance3(first_scale_inblock)
            out = torch.clamp(self.tail(input_pre), 0.0, 1.0)

        return out


if __name__ == '__main__':
    img = torch.randn(1, 305, 128, 128)
    net3 = DeRainNet(inp_chans=305)
    out1 = net3(img)
    print(out1.size())
