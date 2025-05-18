import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange

from .common import WithBiasLayerNorm, BiasFreeLayerNorm
from .freq import FreqQKVGenerator
from .sparse import SparsifyAttention
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
