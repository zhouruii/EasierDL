import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from pytorch_wavelets import DWTForward, DWTInverse, DTCWTForward, DTCWTInverse

from .common import conv3x3
from ..builder import MODULE


class WaveletProj(nn.Module):
    """ DWT, refer to
    https://pytorch-wavelets.readthedocs.io/en/latest/dwt.html
    """

    def __init__(self,
                 in_channels=128,
                 J=1,
                 wave='haar',
                 mode='reflect'):
        super().__init__()

        self.wavelet_transform = DWTForward(J=J, wave=wave, mode=mode)
        self.inverse_transform = DWTInverse(wave=wave)

        self.conv_ll = nn.Sequential(
            conv3x3(in_channels, in_channels, groups=in_channels),
            conv3x3(in_channels, in_channels, groups=in_channels),
        )
        self.conv_lh = nn.Conv2d(in_channels, in_channels, (3, 7), 1, (1, 3), groups=in_channels)
        self.conv_hl = nn.Conv2d(in_channels, in_channels, (7, 3), 1, (3, 1), groups=in_channels)
        self.conv_hh = conv3x3(in_channels, in_channels, groups=in_channels)

    def forward(self, x):
        LL, H = self.wavelet_transform(x)
        LH, HL, HH = H[0][:, :, 0, :, :], H[0][:, :, 1, :, :], H[0][:, :, 2, :, :]

        LL = self.conv_ll(LL)
        LH = self.conv_lh(LH)
        HL = self.conv_hl(HL)
        HH = self.conv_hh(HH)

        H = torch.stack([LH, HL, HH], dim=2)

        out = self.inverse_transform((LL, [H]))  # only support J = 1

        return out


class DTCWTProj(nn.Module):
    """ DTCWT:Dual-Tree Complex Wavelet Transform, refer to
    https://pytorch-wavelets.readthedocs.io/en/latest/dtcwt.html
    """

    def __init__(self,
                 in_channels=128,
                 J=1,
                 biort='legall',
                 qshift='qshift_a'):
        super().__init__()

        self.wavelet_transform = DTCWTForward(J=J, biort=biort, qshift=qshift, mode='symmetric')
        self.inverse_transform = DTCWTInverse(biort=biort, qshift=qshift)

        self.conv_l = nn.Sequential(
            conv3x3(in_channels, in_channels, groups=in_channels),
            conv3x3(in_channels, in_channels, groups=in_channels),
        )
        # 15 45 75 105 135 165
        self.conv_h = nn.ModuleList([conv3x3(in_channels, in_channels, groups=in_channels) for _ in range(6)])

    def forward(self, x):
        # x.shape (B,C,H,W)
        # L.shape (B,C,H,W) J = 1
        # H[0].shape (B,C,6,H,W,2) J = 1, 2 -> real img, 6 -> direction
        L, H = self.wavelet_transform(x)
        L = self.conv_l(L)
        hs = []
        for i in range(6):
            h = H[0][:, :, i, :, :, :]  # (B,C,H,W,2)
            h = rearrange(h, 'b c h w ri -> b (c ri) h w')  # (B,2C,H,W)
            conv = self.conv_h[i]
            h = conv(h)
            h = rearrange(h, 'b (c ri) h w -> b c h w ri', c=L.shape[1], ri=2)  # (B,C,H,W,2)
            hs.append(h)
        H = torch.stack(hs, dim=2)
        out = self.inverse_transform((L, [H]))
        return out


class FreqProj(nn.Module):
    FT_TYPE = ['DWT', 'DTCWT']

    def __init__(self,
                 cfg=None, ):
        super().__init__()
        freq_cfg = cfg.copy()
        freq = freq_cfg.pop('type')
        self.proj = None

        if freq is None:
            raise ValueError(f'param: freq must be given, supported value: {self.FT_TYPE}')
        assert freq in self.FT_TYPE, f'transform : {freq} not supported !'

        if freq == 'DWT':
            self.proj = WaveletProj(**freq_cfg)
        elif freq == 'DTCWT':
            self.proj = DTCWTProj(**freq_cfg)
        else:
            raise NotImplementedError(f'transform : {freq} not supported !')

    def forward(self, x):
        return self.proj(x)


class FreqQKVGenerator(nn.Module):
    def __init__(self,
                 in_channels=128,
                 freq_cfg=None):
        super().__init__()

        self.pw_conv = nn.Conv2d(in_channels, in_channels * 3, 1, 1, 0)
        if freq_cfg is None:
            self.proj_q = conv3x3(in_channels, in_channels, groups=in_channels)
        else:
            self.proj_q = FreqProj(
                cfg=freq_cfg
            )
        self.dw_conv_k = conv3x3(in_channels, in_channels, groups=in_channels)
        self.dw_conv_v = conv3x3(in_channels, in_channels, groups=in_channels)

    def forward(self, x):
        # x.shape (B, C, H, W)
        x = self.pw_conv(x)
        q, k, v = torch.chunk(x, chunks=3, dim=1)

        q = self.proj_q(q)
        k = self.dw_conv_k(k)
        v = self.dw_conv_v(v)

        return q, k, v


@MODULE.register_module()
class DynamicDecoupler(nn.Module):
    """
    refer to SFNet (ICLR 2023)
    """

    def __init__(self, in_channels, kernel_size=3, group=8, norm='BN'):
        super(DynamicDecoupler, self).__init__()
        assert in_channels % group == 0, \
            f'in_channels:{in_channels} must be divided by groups: {group}'
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.group = group

        num_filter_params = group * kernel_size ** 2
        self.conv = nn.Conv2d(in_channels, num_filter_params, kernel_size=1, stride=1, bias=False)

        if norm == 'BN':
            self.norm = nn.BatchNorm2d(num_filter_params)
        elif norm == 'GN':
            self.norm = nn.GroupNorm(group, num_filter_params)
        else:
            raise NotImplementedError(f'norm type: {norm} not supported yet !')
        self.act = nn.Softmax(dim=-2)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.ap_1 = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        identity_input = x
        low_filter1 = self.ap_1(x)
        low_filter = self.conv(low_filter1)
        low_filter = self.norm(low_filter)

        n, c, h, w = x.shape
        # prepare conv
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(n, self.group, c // self.group,
                                                                        self.kernel_size ** 2, h * w)

        n, c1, p, q = low_filter.shape
        low_filter = low_filter.reshape(n, c1 // self.kernel_size ** 2, self.kernel_size ** 2, p * q).unsqueeze(2)
        low_filter = self.act(low_filter)

        # filter conv
        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)

        # inverse
        out_high = identity_input - low_part
        return low_part, out_high


@MODULE.register_module()
class GatedDynamicDecoupler(nn.Module):
    """
    refer to FPro (ECCV 2024)
    """

    def __init__(self, in_channels, kernel_size=3, group=8, norm='BN'):
        super(GatedDynamicDecoupler, self).__init__()
        self.kernel_size = kernel_size
        self.group = group

        num_filter_params = group * kernel_size ** 2
        self.conv = nn.Conv2d(in_channels, num_filter_params, kernel_size=1, stride=1, bias=False)
        self.conv_gate = nn.Conv2d(num_filter_params, num_filter_params, kernel_size=1, stride=1,
                                   bias=False)
        self.act_gate = nn.Sigmoid()
        if norm == 'BN':
            self.norm = nn.BatchNorm2d(num_filter_params)
        elif norm == 'GN':
            self.norm = nn.GroupNorm(group, num_filter_params)
        else:
            raise NotImplementedError(f'norm type: {norm} not supported yet !')
        self.act = nn.Softmax(dim=-2)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

        self.pad = nn.ReflectionPad2d(kernel_size // 2)

        self.ap_1 = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        identity_input = x
        low_filter1 = self.ap_1(x)
        low_filter = self.conv(low_filter1)
        low_filter = low_filter * self.act_gate(self.conv_gate(low_filter))
        low_filter = self.norm(low_filter)

        n, c, h, w = x.shape
        # prepare conv
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(n, self.group, c // self.group,
                                                                        self.kernel_size ** 2, h * w)

        n, c1, p, q = low_filter.shape
        low_filter = low_filter.reshape(n, c1 // self.kernel_size ** 2, self.kernel_size ** 2, p * q).unsqueeze(2)
        low_filter = self.act(low_filter)

        # filter conv
        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)

        # inverse
        out_high = identity_input - low_part
        return low_part, out_high


def chunk_mul(a, b, chunk_size=64):
    output = torch.empty_like(a)
    for i in range(0, a.shape[1], chunk_size):  # 按第1维分块
        output[:, i:i + chunk_size] = a[:, i:i + chunk_size] * b[:, i:i + chunk_size]
    return output


@MODULE.register_module()
class FreqDecouplingReconstruction(nn.Module):
    def __init__(self, d=1, in_channels=128, filter_kernel_size=3, filter_groups=8, norm='BN'):
        super().__init__()
        assert in_channels % filter_groups == 0, \
            f'in_channels:{in_channels} must be divided by filter_groups:{filter_groups}'
        self.d = d
        self.in_channels = in_channels
        self.filter_kernel_size = filter_kernel_size
        self.filter_groups = filter_groups

        self.decoupler = GatedDynamicDecoupler(
            in_channels=self.in_channels,
            kernel_size=self.filter_kernel_size,
            group=self.filter_groups,
            norm=norm
        )
        self.conv_high_freq = nn.Conv2d(in_channels, 1, 1)
        self.gamma = nn.Parameter(torch.tensor(float(1)))

    def forward(self, x, raw):
        # x: B C H W
        low, high = self.decoupler(x)

        # low part
        shortcut = low
        low = chunk_mul(low, raw)  # low * raw
        low = low + raw - shortcut

        # high part
        high = self.conv_high_freq(high)
        tau = torch.exp(self.gamma * self.d)
        high = high * tau

        return low - high


if __name__ == '__main__':
    _inp = torch.randn((2, 64, 128, 128))
    module = GatedDynamicDecoupler(64)
    _out = module(_inp)
