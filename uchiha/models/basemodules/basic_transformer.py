import torch
from torch import nn

from ..builder import BASEMODULE


def _init_vit_weights(m):
    """ ViT weight initialization

    Args:
        m (nn.Module): modules that need to initialize weights
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class FeedForward(nn.Module):
    """ Feedforward-Network

    Args:
        dim (int): input dimension
        hidden_dim (int): dimension of hidden layers
        dropout (float): the rate of `Dropout` layer. Default: 0.0
    """
    def __init__(self, dim, hidden_dim, dropout=0.):

        super().__init__()
        self.net = nn.Sequential(
            # nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """ self attention module

    Args:
        dim (int): input dimension
        num_heads (int): number of heads in `Multi-Head Attention`
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        dropout (float): the rate of `Dropout` layer after qkv calculation. Default: 0.0
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, dropout=0.):

        super().__init__()
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """ basic Transformer block

    Args:
        dim (int): input dimension
        num_heads (int): number of heads in `Multi-Head Attention`
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        mlp_dim (int): dimension of hidden layers in FeedForward
        dropout (float): the rate of `Dropout` layer. Default: 0.0
    """

    def __init__(self, dim, num_heads, qkv_bias, mlp_dim, dropout=0.):

        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, dropout)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


@BASEMODULE.register_module()
class SimpleVisionTransformerLayer(nn.Module):
    """ Vision Transformer (the simplest implementation)

    Args:
        dim (int): input dimension
        depth (int): number of stacked basic-transformer-blocks
        num_heads (int): number of heads in `Multi-Head Attention`
        sequence_length (int): The length of the sequence after changing the shape to (B ,L, C).
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        mlp_ratio (float): ratio of hidden layer to input channel in MLP
        pos (bool): Position encoding will be used if set to True.
        dropout (float): the rate of `Dropout` layer. Default: 0.0
    """

    def __init__(self, dim, depth, num_heads, sequence_length, qkv_bias, mlp_ratio=4., pos=False, dropout=0.):

        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))  # 1*dim
        self.pos = pos
        if pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, sequence_length + 1, dim))  # 3*dim
        self.pos_drop = nn.Dropout(dropout)

        mlp_dim = int(dim * mlp_ratio)
        self.blocks = nn.Sequential(*[
            Block(dim, num_heads, qkv_bias, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

        if self.pos:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # [B, 1, dim]
        x = torch.cat((cls_token, x), dim=1)  # [B, num_patches + 1, dim]

        if self.pos:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x)
        x = self.norm(x)

        return x[:, 0].unsqueeze(1)
