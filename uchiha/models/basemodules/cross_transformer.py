import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange

from ..builder import BASEMODULE


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


@BASEMODULE.register_module()
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
