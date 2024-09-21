import torch
from timm.layers import DropPath, to_2tuple
from einops import rearrange
from torch import nn
import torch.nn.functional as F

from ..builder import BASEMODULE
from ...utils import build_norm





class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, mode='channel'):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.mode = mode
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        q = self.q(x)
        k = self.k(y)
        v = self.v(y)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


@BASEMODULE.register_module()
class CrossTransformer(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(CrossTransformer, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = CrossAttention(dim, num_heads, bias)


    def forward(self, x1, x2):
        # x1 -> q
        # x2 -> k v
        out = self.attn(self.norm1(x1), self.norm2(x2))  # b, c, h, w
        out = x1 + out
        return out