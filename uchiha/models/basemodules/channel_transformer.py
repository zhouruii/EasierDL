import torch
from timm.layers import DropPath, to_2tuple
from torch import nn
import torch.nn.functional as F

from ..builder import BASEMODULE
from ...utils import build_norm


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ChannelAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.v = nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
        """
        B_, N, C = x.shape

        qkv = self.qkv(x)

        qkv = qkv.reshape(B_, N, 3, C).permute(2, 0, 1, 3)
        q = qkv[0].reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = qkv[1].reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = qkv[2]
        v = self.v(v)
        v = v.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ChannelTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        if isinstance(norm_layer, str):
            norm_layer = build_norm(norm_layer)
        self.norm1 = norm_layer(dim)

        self.attn_C = ChannelAttention(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)

        # W-MSA/SW-MSA
        x = self.attn_C(x)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


@BASEMODULE.register_module()
class ChannelTransformerLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_head,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, ):
        super().__init__()
        self.dim = dim
        self.input_resolution = to_2tuple(input_resolution)
        self.depth = depth

        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # build blocks
        self.blocks = nn.ModuleList([
            ChannelTransformerBlock(dim=dim, input_resolution=self.input_resolution,
                                    num_heads=num_head,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=self.dpr[i],
                                    norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        return x


@BASEMODULE.register_module()
class UnetChannelTransformerLayers(nn.Module):
    def __init__(self,
                 dims,
                 input_resolutions,
                 depths,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dims
        self.input_resolution = input_resolutions
        self.depth = depths
        self.num_heads = num_heads
        self.num_layers = len(dims) // 2

        # stochastic depth
        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:len(depths) // 2]))]
        self.dec_dpr = self.dpr[::-1]

        # build layers
        self.layers = nn.ModuleList()

        for i in range(self.num_layers * 2):
            dim = self.dim[i]
            input_resolution = to_2tuple(self.input_resolution[i])
            num_heads = self.num_heads[i]
            drop_path = self.dpr[sum(depths[:i]):sum(depths[:i + 1])]
            if i >= self.num_layers:
                drop_path = self.dec_dpr[sum(depths[self.num_layers:i]):sum(depths[self.num_layers:i + 1])]

            # build layer (blocks)
            layer = nn.Sequential(*[
                ChannelTransformerBlock(dim=dim, input_resolution=input_resolution,
                                        num_heads=num_heads,
                                        mlp_ratio=mlp_ratio,
                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        drop=drop, attn_drop=attn_drop,
                                        drop_path=drop_path[j],
                                        norm_layer=norm_layer)
                for j in range(self.depth[i])])

            self.layers.append(layer)


@BASEMODULE.register_module()
class ChannelTransformerLayers(nn.Module):
    def __init__(self,
                 dims,
                 input_resolutions,
                 depths,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dims
        self.input_resolution = input_resolutions
        self.depth = depths
        self.num_heads = num_heads
        self.num_layers = len(dims)

        # stochastic depth
        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.layers = nn.ModuleList()

        for i in range(self.num_layers):
            dim = self.dim[i]
            input_resolution = to_2tuple(self.input_resolution[i])
            num_heads = self.num_heads[i]
            drop_path = self.dpr[sum(depths[:i]):sum(depths[:i + 1])]

            # build layer (blocks)
            layer = nn.Sequential(*[
                ChannelTransformerBlock(dim=dim, input_resolution=input_resolution,
                                        num_heads=num_heads,
                                        mlp_ratio=mlp_ratio,
                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        drop=drop, attn_drop=attn_drop,
                                        drop_path=drop_path[j],
                                        norm_layer=norm_layer)
                for j in range(self.depth[i])])

            self.layers.append(layer)
