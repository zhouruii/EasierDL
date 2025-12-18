import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from pytorch_wavelets import DWTForward, DWTInverse, DTCWTForward, DTCWTInverse

from .common import conv3x3, GatedUnit
from .cbam import BasicCAM
from ..builder import MODULE


class SimpleWaveletProj(nn.Module):
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


class SimpleDTCWTProj(nn.Module):
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
            conv3x3(in_channels, in_channels, groups=in_channels),
        )
        # 15 45 75 105 135 165
        self.conv_h = nn.ModuleList([nn.Sequential(
            conv3x3(in_channels, in_channels, groups=in_channels),
            conv3x3(in_channels, in_channels, groups=in_channels),
        ) for _ in range(6)])

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


class GatedWaveletProj(nn.Module):
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

        self.gate_ll_rain = GatedUnit(in_channels=in_channels, depth=2, kernel_size=3)
        self.gate_ll_haze = GatedUnit(in_channels=in_channels, depth=2, kernel_size=3)
        self.gate_lh = GatedUnit(in_channels=in_channels, depth=1, kernel_size=(3, 5), padding=(1, 2))
        self.gate_hl = GatedUnit(in_channels=in_channels, depth=1, kernel_size=(5, 3), padding=(2, 1))
        self.gate_hh = GatedUnit(in_channels=in_channels, depth=1, kernel_size=3)

    def forward(self, x, mask):
        LL, H = self.wavelet_transform(x)
        LH, HL, HH = H[0][:, :, 0, :, :], H[0][:, :, 1, :, :], H[0][:, :, 2, :, :]
        mask_rp_ll, mask_rp_lh, mask_rp_hl, mask_rp_hh, mask_hp = mask

        LL_RAIN = self.gate_ll_rain(LL)
        LL_HAZE = self.gate_ll_haze(LL)
        LH = self.gate_lh(LH)
        HL = self.gate_hl(HL)
        HH = self.gate_hh(HH)

        res = LL_RAIN
        LL_RAIN = self.pw_conv1(LL_RAIN)
        LL_RAIN = LL_RAIN * mask_rp_ll
        LL_RAIN = LL_RAIN + res
        res = LL_HAZE
        LL_HAZE = self.pw_conv1(LL_HAZE)
        LL_HAZE = LL_HAZE * mask_rp_ll
        LL_HAZE = LL_HAZE + res
        LL = torch.cat([LL_RAIN, LL_HAZE], dim=1)
        LL = self.cat_conv(LL)
        # --------------- LH --------------- #
        LH = LH * mask_rp_lh
        # --------------- HL --------------- #
        HL = HL * mask_rp_hl
        # --------------- HH --------------- #
        HH = HH * mask_rp_hh

        H = torch.stack([LH, HL, HH], dim=2)

        out = self.inverse_transform((LL, [H]))  # only support J = 1

        return out


class FreqProj(nn.Module):
    FT_TYPE = ['DWT', 'DTCWT', 'RAIN_HAZE_DWT']

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
            self.proj = SimpleWaveletProj(**freq_cfg)
        elif freq == 'DTCWT':
            self.proj = SimpleDTCWTProj(**freq_cfg)
        elif freq == 'RAIN_HAZE_DWT':
            self.proj = GatedWaveletProj(**freq_cfg)
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


class FlexibleGatedDynamicDecoupler(nn.Module):
    """
    Gated Dynamic Decoupler (Flexible Group Version)
    - 支持 in_channels % group != 0
    - 主通道按 group 分组
    - 剩余通道单独处理
    """

    def __init__(self, in_channels, kernel_size=3, group=8, norm='BN'):
        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.group = group
        self.k2 = kernel_size ** 2

        # 主通道 & 剩余通道
        self.main_channels = (in_channels // group) * group
        self.rem_channels = in_channels - self.main_channels

        # === 主分组动态卷积核 ===
        self.num_filter_params_main = group * self.k2
        self.conv_main = nn.Conv2d(in_channels, self.num_filter_params_main, 1, bias=False)
        self.conv_gate_main = nn.Conv2d(self.num_filter_params_main, self.num_filter_params_main, 1, bias=False)

        # === 剩余通道动态卷积核 ===
        if self.rem_channels > 0:
            self.num_filter_params_rem = self.k2
            self.conv_rem = nn.Conv2d(in_channels, self.num_filter_params_rem, 1, bias=False)
            self.conv_gate_rem = nn.Conv2d(self.num_filter_params_rem, self.num_filter_params_rem, 1, bias=False)

        # normalization
        if norm == 'BN':
            self.norm_main = nn.BatchNorm2d(self.num_filter_params_main)
            if self.rem_channels > 0:
                self.norm_rem = nn.BatchNorm2d(self.num_filter_params_rem)
        elif norm == 'GN':
            self.norm_main = nn.GroupNorm(group, self.num_filter_params_main)
            if self.rem_channels > 0:
                self.norm_rem = nn.GroupNorm(1, self.num_filter_params_rem)
        else:
            raise NotImplementedError

        self.act_gate = nn.Sigmoid()
        self.act_kernel = nn.Softmax(dim=-2)

        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        identity = x
        B, C, H, W = x.shape

        # ===============================
        # 1️⃣ 生成动态卷积核（主 + 剩余）
        # ===============================
        pooled = self.pool(x)

        # --- 主分组 ---
        kernel_main = self.conv_main(pooled)
        kernel_main = kernel_main * self.act_gate(self.conv_gate_main(kernel_main))
        kernel_main = self.norm_main(kernel_main)

        kernel_main = kernel_main.reshape(
            B, self.group, self.k2, 1
        )
        kernel_main = self.act_kernel(kernel_main)  # (B, G, K2, 1)

        # --- 剩余 ---
        if self.rem_channels > 0:
            kernel_rem = self.conv_rem(pooled)
            kernel_rem = kernel_rem * self.act_gate(self.conv_gate_rem(kernel_rem))
            kernel_rem = self.norm_rem(kernel_rem)

            kernel_rem = kernel_rem.reshape(B, 1, self.k2, 1)
            kernel_rem = self.act_kernel(kernel_rem)

        # ===============================
        # 2️⃣ unfold 输入
        # ===============================
        x_pad = self.pad(x)
        x_unfold = F.unfold(x_pad, kernel_size=self.kernel_size)
        x_unfold = x_unfold.reshape(B, C, self.k2, H * W)

        # --- 主通道 ---
        x_main = x_unfold[:, :self.main_channels]
        x_main = x_main.reshape(
            B, self.group, self.main_channels // self.group, self.k2, H * W
        )

        out_main = torch.sum(
            x_main * kernel_main.unsqueeze(2),
            dim=3
        ).reshape(B, self.main_channels, H, W)

        # --- 剩余通道 ---
        if self.rem_channels > 0:
            x_rem = x_unfold[:, self.main_channels:].unsqueeze(1)
            out_rem = torch.sum(
                x_rem * kernel_rem.unsqueeze(2),
                dim=3
            ).reshape(B, self.rem_channels, H, W)

            out = torch.cat([out_main, out_rem], dim=1)
        else:
            out = out_main

        # ===============================
        # 3️⃣ 高频部分
        # ===============================
        out_high = identity - out
        return out, out_high


def chunk_mul(a, b, chunk_size=64):
    output = torch.empty_like(a)
    for i in range(0, a.shape[1], chunk_size):  # 按第1维分块
        output[:, i:i + chunk_size] = a[:, i:i + chunk_size] * b[:, i:i + chunk_size]
    return output


@MODULE.register_module()
class FreqDecouplingReconstruction(nn.Module):
    def __init__(self, d=1, in_channels=128, filter_kernel_size=3, filter_groups=8, norm='BN'):
        super().__init__()

        self.d = d
        self.in_channels = in_channels
        self.filter_kernel_size = filter_kernel_size
        self.filter_groups = filter_groups

        if in_channels % filter_groups == 0:
            self.decoupler = GatedDynamicDecoupler(
                in_channels=self.in_channels,
                kernel_size=self.filter_kernel_size,
                group=self.filter_groups,
                norm=norm
            )
        else:
            self.decoupler = FlexibleGatedDynamicDecoupler(
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


@MODULE.register_module()
class ParllelFreqDecouplingReconstruction(nn.Module):
    def __init__(self, d=1.0, in_channels=128, filter_kernel_size=3, filter_groups=8, norm='BN'):
        super().__init__()
        assert in_channels % filter_groups == 0, \
            f'in_channels:{in_channels} must be divided by filter_groups:{filter_groups}'
        self.d = torch.tensor(d)
        self.in_channels = in_channels
        self.filter_kernel_size = filter_kernel_size
        self.filter_groups = filter_groups

        self.cam = BasicCAM(in_channel=in_channels, out_channel=in_channels)
        self.dw_conv = conv3x3(in_channels, in_channels, groups=in_channels)

        self.low_filter = GatedDynamicDecoupler(
            in_channels=self.in_channels,
            kernel_size=self.filter_kernel_size,
            group=self.filter_groups,
            norm=norm
        )
        self.high_filter = GatedDynamicDecoupler(
            in_channels=self.in_channels,
            kernel_size=self.filter_kernel_size,
            group=self.filter_groups,
            norm=norm
        )

        self.gate_low = GatedUnit(in_channels, depth=2)
        self.gate_high = GatedUnit(in_channels, depth=1)

        self.pw_conv_low = nn.Conv2d(in_channels, in_channels, 1)
        self.pw_conv_high = nn.Conv2d(in_channels, 1, 1)
        self.gamma = nn.Parameter(torch.tensor(float(1)))

    def forward(self, x, raw):
        low = x
        high = x
        low = self.cam(low)
        high = self.dw_conv(high)
        # x: B C H W
        low, _ = self.low_filter(low)
        _, high = self.high_filter(high)

        # low part
        low = self.gate_low(low)
        low = self.pw_conv_low(low)
        shortcut = low
        low = chunk_mul(low, raw)  # low * raw
        low = low + raw - shortcut

        # high part
        high = self.gate_high(high)
        high = self.pw_conv_high(high)
        tau = torch.exp(self.gamma * self.d)
        high = high * tau

        return low - high


if __name__ == '__main__':
    _inp = torch.randn((2, 64, 128, 128))
    module = GatedDynamicDecoupler(64)
    _out = module(_inp)
