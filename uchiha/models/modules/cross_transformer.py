import torch
from pytorch_wavelets import DWTForward, DWTInverse
from torch import nn
import torch.nn.functional as F

from einops import rearrange

from .common import WithBiasLayerNorm, BiasFreeLayerNorm, conv3x3, GatedUnit
from .freq import FreqQKVGenerator
from .sparse import SparsifyAttention
from .swin_transformer import window_partition, window_reverse
from ..builder import MODULE, build_module


class SpatialUniCrossAttention(nn.Module):
    """ Cross-Attention between the primary and secondary

    Args:
        input_dim (int): Number of input channels.
        num_heads (int): Number of heads in `Multi-Head Attention`
        bias (bool): If True, add a learnable bias to query, key, value and projection
    """

    def __init__(self,
                 input_dim=256,
                 num_heads=8,
                 bias=False
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Linear(input_dim, input_dim, bias=bias)
        self.kv = nn.Linear(input_dim, input_dim * 2, bias=bias)

        self.output_proj = nn.Linear(input_dim, input_dim, bias=bias)

    def forward(self, x, y):
        if len(x.shape) == 4:
            _, _, H, W = x.shape
            x = rearrange(x, 'b c h w -> b (h w) c')
            y = rearrange(y, 'b c h w -> b (h w) c')
            spatial_mode = True
        else:
            spatial_mode = False
            H, W = None, None

        q = self.q(x)
        k, v = torch.split(self.kv(y), [self.input_dim, self.input_dim], dim=-1)

        q = rearrange(q, 'b l (head c) -> b head l c', head=self.num_heads)
        k = rearrange(k, 'b l (head c) -> b head l c', head=self.num_heads)
        v = rearrange(v, 'b l (head c) -> b head l c', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = rearrange(out, 'b head l c -> b l (head c)', head=self.num_heads)
        out = self.output_proj(out)
        return rearrange(out, 'b (h w) c -> b c h w', h=H, w=W) if spatial_mode else out


@MODULE.register_module()
class SpatialCrossAttentionBlock(nn.Module):
    """ CrossAttention with LayerNorm

    Args:
        input_dim (int): Number of input channels.
        num_heads (int): Number of heads in `Multi-Head Attention`
        bias (bool): If True, add a learnable bias to query, key, value and projection. Default: False
        mode (str): If set to channel, cross attention calculations will be performed on the channel, else spatial
    """

    def __init__(self,
                 input_dim,
                 num_heads=4,
                 bias=False,
                 mode='spatial'
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.mode = mode

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        if self.mode == 'spatial':
            self.attn = SpatialUniCrossAttention(
                input_dim=self.input_dim,
                num_heads=self.num_heads,
                bias=bias
            )

    def forward(self, x1, x2):
        out = self.attn(self.norm1(x1), self.norm2(x2))
        out = x1 + out
        return out


class ChannelUniCrossAttention(nn.Module):
    def __init__(self,
                 seq_len=96,
                 factor=2.0,
                 bias=False
                 ):
        super().__init__()
        self.seq_len = seq_len

        self.inner_len = int(self.seq_len * factor)

        self.q = nn.Linear(self.seq_len, self.inner_len, bias=bias)
        self.kv = nn.Linear(self.seq_len, self.inner_len + self.seq_len, bias=bias)

        self.output_proj = nn.Linear(self.seq_len, self.seq_len, bias=bias)

    def forward(self, x, y):
        if len(x.shape) == 4:
            _, _, H, W = x.shape
            x = rearrange(x, 'b c h w -> b c (h w)')
            y = rearrange(y, 'b c h w -> b c (h w)')
            spatial_mode = True
        else:
            spatial_mode = False
            H, W = None, None

        q = self.q(x)
        k, v = torch.split(self.kv(y), [self.inner_len, self.seq_len], dim=-1)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = self.output_proj(out)
        return rearrange(out, 'b c (h w) -> b c h w', h=H, w=W) if spatial_mode else out


class SelfCrossAttention(nn.Module):
    def __init__(self,
                 in_channels=128,
                 num_heads=8,
                 prior_cfg=None,
                 freq_cfg=None,
                 sparse_strategy=None):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.use_prior = prior_cfg is not None
        assert in_channels % num_heads == 0, f'in_channels:{in_channels} must be divided by num_heads:{num_heads}'

        head_dim = in_channels // num_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5))

        self.qkv_generator = FreqQKVGenerator(in_channels=in_channels, freq_cfg=freq_cfg)
        self.sparse_opt = SparsifyAttention(sparse_strategy)
        if self.use_prior:
            self.prior_opt = build_module(prior_cfg)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x, prior=None):
        # x.shape (B,C,H,W)
        _, _, H, W = x.shape

        q, k, v = self.qkv_generator(x)
        q = rearrange(q, 'b (nh c) h w -> b nh c (h w)', h=H, w=W, nh=self.num_heads)
        k = rearrange(k, 'b (nh c) h w -> b nh c (h w)', h=H, w=W, nh=self.num_heads)
        v = rearrange(v, 'b (nh c) h w -> b nh c (h w)', h=H, w=W, nh=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        att = (q @ k.transpose(-2, -1)) * self.scale
        att = self.sparse_opt(att)  # (C, C)

        out = (att @ v)
        if self.use_prior and prior is not None:
            prior, mask = self.prior_opt(prior)
            v = rearrange(v, 'b nh c (h w) -> b nh c h w', h=H, w=W)
            v = v * mask
            out += rearrange(v, 'b nh c h w -> b nh c (h w)', h=H, w=W)

        out = rearrange(out, 'b nh c (h w) -> b (nh c) h w', nh=self.num_heads, h=H, w=W)

        return self.proj_out(out), prior


@MODULE.register_module()
class SelfCrossAttentionBlock(nn.Module):
    def __init__(self,
                 in_channels=128,
                 num_heads=8,
                 prior_cfg=False,
                 freq_cfg=None,
                 ffn_cfg=None,
                 sparse_strategy=None,
                 ln_bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.use_prior = prior_cfg is not None
        self.ln_bias = ln_bias

        self.attn = SelfCrossAttention(
            in_channels=in_channels,
            num_heads=num_heads,
            prior_cfg=prior_cfg,
            freq_cfg=freq_cfg,
            sparse_strategy=sparse_strategy
        )

        self.ffn = build_module(ffn_cfg)

        self.ln1 = WithBiasLayerNorm(in_channels) if self.ln_bias else BiasFreeLayerNorm(in_channels)
        self.ln2 = WithBiasLayerNorm(in_channels) if self.ln_bias else BiasFreeLayerNorm(in_channels)

    def forward(self, x, prior=None):
        # x.shape (B,C,H,W)
        _, _, H, W = x.shape

        shortcut = x
        x = self.ln1(x)
        x, prior = self.attn(x, prior)
        x = x + shortcut

        shortcut = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = x + shortcut

        return x, prior


@MODULE.register_module()
class SelfCrossAttentionLayer(nn.Module):
    def __init__(self,
                 in_channels=128,
                 num_heads=8,
                 num_blocks=4,
                 freq_cfg=None,
                 prior_cfg=None,  # forward for prior
                 ffn_cfg=None,
                 sparse_strategy=None,
                 ln_bias=True,  # bias for LayerNorm
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        self.blocks = nn.ModuleList([
            SelfCrossAttentionBlock(
                in_channels=in_channels,
                num_heads=num_heads,
                prior_cfg=prior_cfg,
                freq_cfg=freq_cfg,
                ffn_cfg=ffn_cfg,
                sparse_strategy=sparse_strategy,
                ln_bias=ln_bias
            ) for _ in range(num_blocks)])

    def forward(self, x, prior=None):
        # x.shape (B,C,H,W)
        for blk in self.blocks:
            x, prior = blk(x, prior)

        return x, prior


class WaveletHeterogeneousAttention(nn.Module):
    def __init__(self,
                 in_channels=128,
                 num_heads=8,
                 window_size=4,
                 prior_cfg=None,
                 sparse_strategy=None):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.window_size = window_size
        self.sparse_strategy = sparse_strategy
        self.wavelet_transform = DWTForward(J=1, wave='haar', mode='reflect')
        self.inverse_transform = DWTInverse(wave='haar')

        assert in_channels % num_heads == 0, f'in_channels:{in_channels} must be divided by num_heads:{num_heads}'
        self.head_dim = in_channels // num_heads

        # ------------------------------------------ Projection ------------------------------------------ #
        self.linear_proj = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1, padding=0)
        self.q_proj = conv3x3(in_channels, in_channels, groups=in_channels)
        self.k_proj = nn.ModuleList([conv3x3(in_channels, in_channels, groups=in_channels, stride=2) for _ in range(4)])
        self.v_proj = nn.ModuleList([conv3x3(in_channels, in_channels, groups=in_channels, stride=2) for _ in range(4)])

        # ------------------------------------------ Gate Unit ------------------------------------------ #
        self.gate_ll_rain = GatedUnit(in_channels=in_channels, depth=2, kernel_size=3)
        self.gate_ll_haze = GatedUnit(in_channels=in_channels, depth=2, kernel_size=3)
        self.gate_lh = GatedUnit(in_channels=in_channels, depth=1, kernel_size=(3, 5), padding=(1, 2))
        self.gate_hl = GatedUnit(in_channels=in_channels, depth=1, kernel_size=(5, 3), padding=(2, 1))
        self.gate_hh = GatedUnit(in_channels=in_channels, depth=1, kernel_size=3)

        # ------------------------------------------ Prior Unit ------------------------------------------ #
        self.prior_opt = build_module(prior_cfg)
        self.pw_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.pw_conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.cat_conv = nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=1)

        # ------------------------------------------ Attn Unit ------------------------------------------ #
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.scale = self.head_dim ** -0.5
        self.sparse_opt_ll = SparsifyAttention(sparse_strategy) if self.sparse_strategy else nn.Softmax(dim=-1)
        self.sparse_opt_lh = SparsifyAttention(sparse_strategy)
        self.sparse_opt_hl = SparsifyAttention(sparse_strategy)
        self.sparse_opt_hh = SparsifyAttention(sparse_strategy)
        self.w1 = nn.Parameter(torch.tensor(0.5))
        self.w2 = nn.Parameter(torch.tensor(0.2))
        self.w3 = nn.Parameter(torch.tensor(0.2))
        self.w4 = nn.Parameter(torch.tensor(0.1))

        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x, prior=None):
        # x.shape (B,C,H,W)
        # ------------------------------------------ Projection ------------------------------------------ #
        _, _, H, W = x.shape
        x = self.linear_proj(x)
        q, _k, _v = torch.chunk(x, chunks=3, dim=1)
        q = self.q_proj(q)
        k_ll, k_lh, k_hl, k_hh = [proj(_k) for proj in self.k_proj]
        v_ll, v_lh, v_hl, v_hh = [proj(_v) for proj in self.v_proj]

        # ------------------------------------------ Gate Unit ------------------------------------------ #
        LL, H = self.wavelet_transform(q)
        LH, HL, HH = H[0][:, :, 0, :, :], H[0][:, :, 1, :, :], H[0][:, :, 2, :, :]
        q_ll_rain = self.gate_ll_rain(LL)
        q_ll_haze = self.gate_ll_haze(LL)
        q_lh = self.gate_lh(LH)
        q_hl = self.gate_hl(HL)
        q_hh = self.gate_hh(HH)

        # ------------------------------------------ Prior Unit ------------------------------------------ #
        # --------------- prior --------------- #
        prior, mask = self.prior_opt(prior)
        mask_gp, mask_rp_ll, mask_rp_lh, mask_rp_hl, mask_hp = mask
        # --------------- LL --------------- #
        res = q_ll_rain
        q_ll_rain = self.pw_conv1(q_ll_rain)
        q_ll_rain = q_ll_rain * mask_rp_ll
        q_ll_rain = q_ll_rain + res
        res = q_ll_haze
        q_ll_haze = self.pw_conv1(q_ll_haze)
        q_ll_haze = q_ll_haze * mask_hp
        q_ll_haze = q_ll_haze + res
        q_ll = torch.cat([q_ll_rain, q_ll_haze], dim=1)
        q_ll = self.cat_conv(q_ll)
        # --------------- LH --------------- #
        q_lh = q_lh * mask_rp_lh
        # --------------- HL --------------- #
        q_hl = q_hl * mask_rp_hl

        # ------------------------------------------ Attn Unit ------------------------------------------ #
        out_ll = self.cal_self_attn(q_ll, k_ll, v_ll)
        out_lh = self.cal_win_attn(q_lh, k_lh, v_lh, self.window_size, sub_band='lh')
        out_hl = self.cal_win_attn(q_hl, k_hl, v_hl, self.window_size, sub_band='hl')
        out_hh = self.cal_win_attn(q_hh, k_hh, v_hh, self.window_size, sub_band='hh')
        out = self.inverse_transform((out_ll, [torch.stack([out_lh, out_hl, out_hh], dim=2)]))

        res = _v
        res = res * mask_gp
        # v = rearrange(v, 'b (nh c) h w -> b nh c (h w)', nh=self.num_heads)
        # out = (attn @ v)
        # out = rearrange(out, 'b nh c (h w) -> b (nh c) h w', h=H, w=W, nh=self.num_heads)
        out = out + res

        return self.proj_out(out), prior

    def cal_self_attn(self, q, k, v):
        B, C, H, W = q.shape
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature  # B
        attn = self.sparse_opt_ll(attn)

        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)

        return out

    def cal_win_attn(self, q, k, v, window_size, sub_band='lh'):
        B, C, H, W = q.shape
        q = rearrange(q, 'b c h w -> b h w c')
        k = rearrange(k, 'b c h w -> b h w c')
        v = rearrange(v, 'b c h w -> b h w c')
        q_windows = rearrange(window_partition(q, window_size), 'nW_B wW wH (head c) -> nW_B head (wW wH) c',
                              head=self.num_heads)
        k_windows = rearrange(window_partition(k, window_size), 'nW_B wW wH (head c) -> nW_B head (wW wH) c',
                              head=self.num_heads)
        v_windows = rearrange(window_partition(v, window_size), 'nW_B wW wH (head c) -> nW_B head (wW wH) c',
                              head=self.num_heads)

        attn_windows = (q_windows @ k_windows.transpose(-2, -1)) * self.scale
        if sub_band == 'lh':
            sparse_opt = self.sparse_opt_lh
        elif sub_band == 'hl':
            sparse_opt = self.sparse_opt_hl
        elif sub_band == 'hh':
            sparse_opt = self.sparse_opt_hh
        else:
            raise NotImplementedError(f'sparse for sub-band:{sub_band} not supported yet!')
        attn_windows = sparse_opt(attn_windows)
        out = attn_windows @ v_windows  # B nH N C

        out = rearrange(out, 'nW_B head (wW wH) C -> nW_B wW wH (head C)', wW=window_size)
        out = rearrange(window_reverse(out, window_size, H, W), 'b h w c -> b c h w')

        return out


@MODULE.register_module()
class WaveletHeterogeneousAttentionBlock(nn.Module):
    def __init__(self,
                 in_channels=128,
                 num_heads=8,
                 window_size=4,
                 prior_cfg=False,
                 ffn_cfg=None,
                 sparse_strategy=None,
                 ln_bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.ln_bias = ln_bias

        self.attn = WaveletHeterogeneousAttention(
            in_channels=in_channels,
            num_heads=num_heads,
            window_size=window_size,
            prior_cfg=prior_cfg,
            sparse_strategy=sparse_strategy
        )

        self.ffn = build_module(ffn_cfg)

        self.ln1 = WithBiasLayerNorm(in_channels) if self.ln_bias else BiasFreeLayerNorm(in_channels)
        self.ln2 = WithBiasLayerNorm(in_channels) if self.ln_bias else BiasFreeLayerNorm(in_channels)

    def forward(self, x, prior=None):
        # x.shape (B,C,H,W)
        _, _, H, W = x.shape

        shortcut = x
        x = self.ln1(x)
        x, prior = self.attn(x, prior)
        x = x + shortcut

        shortcut = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = x + shortcut

        return x, prior


@MODULE.register_module()
class WaveletHeterogeneousAttentionLayer(nn.Module):
    def __init__(self,
                 in_channels=128,
                 num_heads=8,
                 window_size=4,
                 num_blocks=4,
                 prior_cfg=None,  # forward for prior
                 ffn_cfg=None,
                 sparse_strategy=None,
                 ln_bias=True,  # bias for LayerNorm
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        self.blocks = nn.ModuleList([
            WaveletHeterogeneousAttentionBlock(
                in_channels=in_channels,
                num_heads=num_heads,
                window_size=window_size,
                prior_cfg=prior_cfg,
                ffn_cfg=ffn_cfg,
                sparse_strategy=sparse_strategy,
                ln_bias=ln_bias
            ) for _ in range(num_blocks)])

    def forward(self, x, prior=None):
        # x.shape (B,C,H,W)
        for blk in self.blocks:
            x, prior = blk(x, prior)

        return x, prior


class WaveletFreqWiseAttention(nn.Module):
    def __init__(self,
                 in_channels=128,
                 num_heads=8,
                 prior_cfg=None,
                 sparse_strategy=None):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.sparse_strategy = sparse_strategy
        self.wavelet_transform = DWTForward(J=1, wave='haar', mode='reflect')
        self.inverse_transform = DWTInverse(wave='haar')

        assert in_channels % num_heads == 0, f'in_channels:{in_channels} must be divided by num_heads:{num_heads}'
        self.head_dim = in_channels // num_heads

        # ------------------------------------------ Projection ------------------------------------------ #
        self.linear_proj = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1, padding=0)
        self.q_proj = conv3x3(in_channels, in_channels, groups=in_channels)
        self.k_proj = conv3x3(in_channels, in_channels, groups=in_channels)
        self.v_proj = conv3x3(in_channels, in_channels, groups=in_channels)

        # ------------------------------------------ Gate Unit ------------------------------------------ #
        self.gate_ll_rain = GatedUnit(in_channels=in_channels, depth=2, kernel_size=3)
        self.gate_ll_haze = GatedUnit(in_channels=in_channels, depth=2, kernel_size=3)
        self.gate_lh = GatedUnit(in_channels=in_channels, depth=1, kernel_size=(3, 5), padding=(1, 2))
        self.gate_hl = GatedUnit(in_channels=in_channels, depth=1, kernel_size=(5, 3), padding=(2, 1))
        self.gate_hh = GatedUnit(in_channels=in_channels, depth=1, kernel_size=3)

        # ------------------------------------------ Prior Unit ------------------------------------------ #
        self.prior_opt = build_module(prior_cfg)
        self.pw_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.pw_conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.cat_conv = nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=1)

        # ------------------------------------------ Attn Unit ------------------------------------------ #
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.sparse_opt = SparsifyAttention(sparse_strategy)

        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x, prior=None):
        # x.shape (B,C,H,W)
        # ------------------------------------------ Projection ------------------------------------------ #
        _, _, H, W = x.shape
        x = self.linear_proj(x)
        q, k, v = torch.chunk(x, chunks=3, dim=1)
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # ------------------------------------------ Gate Unit ------------------------------------------ #
        q_ll, q_h = self.wavelet_transform(q)
        LH, HL, HH = q_h[0][:, :, 0, :, :], q_h[0][:, :, 1, :, :], q_h[0][:, :, 2, :, :]
        q_ll_rain = self.gate_ll_rain(q_ll)
        q_ll_haze = self.gate_ll_haze(q_ll)
        q_lh = self.gate_lh(LH)
        q_hl = self.gate_hl(HL)
        q_hh = self.gate_hh(HH)

        # ------------------------------------------ Prior Unit ------------------------------------------ #
        # --------------- prior --------------- #
        prior, mask = self.prior_opt(prior)
        mask_rp_ll, mask_rp_lh, mask_rp_hl, mask_rp_hh, mask_hp = mask
        # --------------- LL --------------- #
        res = q_ll_rain
        q_ll_rain = self.pw_conv1(q_ll_rain)
        q_ll_rain = q_ll_rain * mask_rp_ll
        q_ll_rain = q_ll_rain + res
        res = q_ll_haze
        q_ll_haze = self.pw_conv2(q_ll_haze)
        q_ll_haze = q_ll_haze * mask_hp
        q_ll_haze = q_ll_haze + res
        q_ll = torch.cat([q_ll_rain, q_ll_haze], dim=1)
        q_ll = self.cat_conv(q_ll)
        # --------------- LH --------------- #
        q_lh = q_lh * mask_rp_lh
        # --------------- HL --------------- #
        q_hl = q_hl * mask_rp_hl
        # --------------- HH --------------- #
        q_hh = q_hh * mask_rp_hh

        # ------------------------------------------ IDWT ------------------------------------------ #
        q = self.inverse_transform((q_ll, [torch.stack([q_lh, q_hl, q_hh], dim=2)]))

        q = rearrange(q, 'b (nh c) h w -> b nh c (h w)', h=H, w=W, nh=self.num_heads)
        k = rearrange(k, 'b (nh c) h w -> b nh c (h w)', h=H, w=W, nh=self.num_heads)
        v = rearrange(v, 'b (nh c) h w -> b nh c (h w)', h=H, w=W, nh=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        att = (q @ k.transpose(-2, -1)) * self.temperature
        att = self.sparse_opt(att)  # (C, C)

        out = (att @ v)
        out = rearrange(out, 'b nh c (h w) -> b (nh c) h w', nh=self.num_heads, h=H, w=W)

        return self.proj_out(out), prior


@MODULE.register_module()
class WaveletFreqWiseAttentionBlock(nn.Module):
    def __init__(self,
                 in_channels=128,
                 num_heads=8,
                 prior_cfg=False,
                 ffn_cfg=None,
                 sparse_strategy=None,
                 ln_bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.use_prior = prior_cfg is not None
        self.ln_bias = ln_bias

        self.attn = WaveletFreqWiseAttention(
            in_channels=in_channels,
            num_heads=num_heads,
            prior_cfg=prior_cfg,
            sparse_strategy=sparse_strategy
        )

        self.ffn = build_module(ffn_cfg)

        self.ln1 = WithBiasLayerNorm(in_channels) if self.ln_bias else BiasFreeLayerNorm(in_channels)
        self.ln2 = WithBiasLayerNorm(in_channels) if self.ln_bias else BiasFreeLayerNorm(in_channels)

    def forward(self, x, prior=None):
        # x.shape (B,C,H,W)
        _, _, H, W = x.shape

        shortcut = x
        x = self.ln1(x)
        x, prior = self.attn(x, prior)
        x = x + shortcut

        shortcut = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = x + shortcut

        return x, prior


@MODULE.register_module()
class WaveletFreqWiseAttentionLayer(nn.Module):
    def __init__(self,
                 in_channels=128,
                 num_heads=8,
                 num_blocks=4,
                 prior_cfg=None,  # forward for prior
                 ffn_cfg=None,
                 sparse_strategy=None,
                 ln_bias=True,  # bias for LayerNorm
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        self.blocks = nn.ModuleList([
            WaveletFreqWiseAttentionBlock(
                in_channels=in_channels,
                num_heads=num_heads,
                prior_cfg=prior_cfg,
                ffn_cfg=ffn_cfg,
                sparse_strategy=sparse_strategy,
                ln_bias=ln_bias
            ) for _ in range(num_blocks)])

    def forward(self, x, prior=None):
        # x.shape (B,C,H,W)
        for blk in self.blocks:
            x, prior = blk(x, prior)

        return x, prior
