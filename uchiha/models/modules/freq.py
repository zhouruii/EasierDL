import torch
from torch import nn
from einops import rearrange

from pytorch_wavelets import DWTForward, DWTInverse, DTCWTForward, DTCWTInverse

from uchiha.models.modules.common import conv3x3


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
                 cfg=None,):
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
