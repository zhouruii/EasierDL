import torch
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple
from torch import nn

from ..builder import BASEMODULE, build_downsample
from ...utils.model import build_norm, build_act, cfg_decomposition


class Mlp(nn.Module):
    """ MLP in Channel Transformer

    Args:
        in_features (int): number of input channels
        hidden_features (int): number of hidden layer channels
        out_features (int): number of output channels
        act_layer (nn.Module): activation function
        drop (float): the rate of `Dropout` layer. Default: 0.0
    """

    def __init__(self, in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class ChannelAttention(nn.Module):
    r""" attention operations at the channel

    draw inspiration from Window based multi-head self attention (W-MSA)

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of heads in `Multi-Head Attention`
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
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
            x (Tensor):: input features with shape of (num_windows*B, N, C)
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

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ChannelTransformerBlock(nn.Module):
    """ The block containing Channel Attention, norm and MLP

    DropPath: Randomly dropout the entire path, usually at the network structure level,
    dropout a computational path, such as the residual path of the network or some sub-module in the Transformer.

    Args:
        dim (int): Number of input channels.
        input_resolution (Tuple(int)): spatial resolution of input features
        num_heads (int): Number of heads in `Multi-Head Attention`
        mlp_ratio (float): ratio of hidden layer to input channel in MLP
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        drop (float): Dropout ratio of output of AttentionBlock. Default: 0.0
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        drop_path (float): Dropout ratio of entire path
        act_layer (nn.Module): activation function
        norm_layer (nn.Module): normalization layer before Attention and MLP. Default: None
    """

    def __init__(self, dim,
                 input_resolution,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)

        self.attn_C = ChannelAttention(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path: nn.Module = DropPath(drop_path) if drop_path > 0. else nn.Identity()
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
    """ Stacked Channel-Transformer-Block

    Args:
        dim (int): Number of input channels.
        input_resolution (Tuple(int) | int): spatial resolution of input features
        depth (int): number of stacked channel-transformer-blocks
        num_heads (int): Number of heads in `Multi-Head Attention`
        mlp_ratio (float): ratio of hidden layer to input channel in MLP
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        drop_rate (): Dropout ratio of output of Attention. Default: 0.0
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        drop_path (List | float): The probability of DropPath of each `ChannelTransformer Block`.
        act_layer (nn.Module | str): activation function in MLP.
        norm_layer (nn.Module | str): normalization layer before Attention and MLP. Default: None
    """

    def __init__(self, dim,
                 input_resolution,
                 depth,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 downsample=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = to_2tuple(input_resolution) if isinstance(input_resolution, int) else input_resolution
        self.depth = depth

        self.downsample = build_downsample(downsample)

        if isinstance(norm_layer, str):
            norm_layer = build_norm(norm_layer)

        if isinstance(act_layer, str):
            act_layer = build_act(act_layer)

        # build blocks
        self.blocks = nn.ModuleList([
            ChannelTransformerBlock(dim=dim, input_resolution=self.input_resolution,
                                    num_heads=num_heads,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop_rate, attn_drop=attn_drop,
                                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                    act_layer=act_layer,
                                    norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        out = self.downsample(x) if self.downsample else x

        return out


@BASEMODULE.register_module()
class ChannelTransformerLayers(nn.Module):
    """ Collection of Channel-Transformer-Layers

    Args:
        dims (List[int]): Number of input channels.
        input_resolutions (List[int | Tuple(int)]): spatial resolution of input features
        depths (List[int]): number of stacked channel-transformer-blocks
        num_heads (List[int]): Number of heads in `Multi-Head Attention`
        mlp_ratio (float): ratio of hidden layer to input channel in MLP
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        drop_rate (): Dropout ratio of output of Attention. Default: 0.0
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        drop_path_rate (float): Rate required to generate drop path, it will be called by
            `torch.linspace(0, drop_path_rate, depth)`
            designed to increase the probability of DropPath as the depth increases
        act_layer (nn.Module | str): activation function in MLP.
        norm_layer (nn.Module | str): normalization layer before Attention and MLP. Default: None
    """

    def __init__(self,
                 dims,
                 input_resolutions,
                 depths,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop=0.,
                 drop_path_rate=0.1,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 downsamples=None):
        super().__init__()
        self.dims = dims
        self.input_resolutions = input_resolutions
        self.depths = depths
        self.num_heads = num_heads
        self.num_layers = len(dims)

        self.downsamples = cfg_decomposition(downsamples)
        if len(self.downsamples) < self.num_layers:
            self.downsamples.extend([None] * (self.num_layers - len(self.downsamples)))

        # stochastic depth
        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.layers = nn.ModuleList()

        for i in range(self.num_layers):
            dim = self.dims[i]
            input_resolution = self.input_resolutions[i]
            num_heads = self.num_heads[i]
            depth = self.depths[i]
            downsample = self.downsamples[i]

            drop_path = self.dpr[sum(depths[:i]):sum(depths[:i + 1])]

            # build layer
            layer = ChannelTransformerLayer(dim=dim, input_resolution=input_resolution,
                                            depth=depth, num_heads=num_heads,
                                            mlp_ratio=mlp_ratio,
                                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop_rate=drop_rate, attn_drop=attn_drop,
                                            drop_path=drop_path,
                                            act_layer=act_layer,
                                            norm_layer=norm_layer,
                                            downsample=downsample)

            self.layers.append(layer)


@BASEMODULE.register_module()
class ChannelTransformerLayerList(nn.ModuleList):
    """ Collection of Channel-Transformer-Layers

    Args:
        dims (List[int]): Number of input channels.
        input_resolutions (List[int | Tuple(int)]): spatial resolution of input features
        depths (List[int]): number of stacked channel-transformer-blocks
        num_heads (List[int]): Number of heads in `Multi-Head Attention`
        mlp_ratio (float): ratio of hidden layer to input channel in MLP
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        drop_rate (): Dropout ratio of output of Attention. Default: 0.0
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        drop_path_rate (float): Rate required to generate drop path, it will be called by
            `torch.linspace(0, drop_path_rate, depth)`
            designed to increase the probability of DropPath as the depth increases
        act_layer (nn.Module | str): activation function in MLP.
        norm_layer (nn.Module | str): normalization layer before Attention and MLP. Default: None
    """

    def __init__(self,
                 dims,
                 input_resolutions,
                 depths,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop=0.,
                 drop_path_rate=0.1,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 downsamples=None):
        super().__init__()
        self.dims = dims
        self.input_resolutions = input_resolutions
        self.depths = depths
        self.num_heads = num_heads
        self.num_layers = len(dims)

        self.downsamples = cfg_decomposition(downsamples)
        if not self.downsamples:
            self.downsamples = [None] * self.num_layers
        if len(self.downsamples) < self.num_layers:
            self.downsamples.extend([None] * (self.num_layers - len(self.downsamples)))

        # stochastic depth
        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        layers = []

        for i in range(self.num_layers):
            dim = self.dims[i]
            input_resolution = self.input_resolutions[i]
            num_heads = self.num_heads[i]
            depth = self.depths[i]
            downsample = self.downsamples[i]

            drop_path = self.dpr[sum(depths[:i]):sum(depths[:i + 1])]

            # build layer
            layer = ChannelTransformerLayer(dim=dim, input_resolution=input_resolution,
                                            depth=depth, num_heads=num_heads,
                                            mlp_ratio=mlp_ratio,
                                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop_rate=drop_rate, attn_drop=attn_drop,
                                            drop_path=drop_path,
                                            act_layer=act_layer,
                                            norm_layer=norm_layer,
                                            downsample=downsample)

            layers.append(layer)

        self.extend(layers)


@BASEMODULE.register_module()
class UnetChannelTransformerLayers(nn.Module):
    """ Collection of Channel-Transformer-Layers in the shape of unet

    Args:
        dims (List[int]): Number of input channels.
        input_resolutions (List[int]): spatial resolution of input features
        depths (List[int]): number of stacked channel-transformer-blocks
        num_heads (List[int]): Number of heads in `Multi-Head Attention`
        mlp_ratio (float): ratio of hidden layer to input channel in MLP
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        drop_rate (): Dropout ratio of output of Attention. Default: 0.0
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        drop_path_rate (float): Rate required to generate drop path, it will be called by
            `torch.linspace(0, drop_path_rate, depth)`
            designed to increase the probability of DropPath as the depth increases
        act_layer (nn.Module | str): activation function in MLP.
        norm_layer (nn.Module | str): normalization layer before Attention and MLP. Default: None
    """

    def __init__(self,
                 dims,
                 input_resolutions,
                 depths,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop=0.,
                 drop_path_rate=0.1,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 downsamples=None):
        super().__init__()
        self.dims = dims
        self.input_resolutions = input_resolutions
        self.depths = depths
        self.num_heads = num_heads
        self.num_layers = len(dims) // 2

        self.downsamples = cfg_decomposition(downsamples)
        if len(self.downsamples) < self.num_layers * 2:
            self.downsamples.extend([None] * (self.num_layers * 2 - len(self.downsamples)))

        # stochastic depth
        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:len(depths) // 2]))]
        self.dec_dpr = self.dpr[::-1]

        # build layers
        self.layers = nn.ModuleList()

        for i in range(self.num_layers * 2):
            dim = self.dims[i]
            input_resolution = self.input_resolutions[i]
            depth = self.depths[i]
            num_heads = self.num_heads[i]
            downsample = self.downsamples[i]

            drop_path = self.dpr[sum(depths[:i]):sum(depths[:i + 1])]
            if i >= self.num_layers:
                drop_path = self.dec_dpr[sum(depths[self.num_layers:i]):sum(depths[self.num_layers:i + 1])]

            # build layer
            layer = ChannelTransformerLayer(dim=dim, input_resolution=input_resolution,
                                            depth=depth, num_heads=num_heads,
                                            mlp_ratio=mlp_ratio,
                                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop_rate=drop_rate, attn_drop=attn_drop,
                                            drop_path=drop_path,
                                            act_layer=act_layer,
                                            norm_layer=norm_layer,
                                            downsample=downsample)

            self.layers.append(layer)


@BASEMODULE.register_module()
class UnetChannelTransformerLayerList(nn.Module):
    """ Collection of Channel-Transformer-Layers in the shape of unet

    Args:
        dims (List[int]): Number of input channels.
        input_resolutions (List[int]): spatial resolution of input features
        depths (List[int]): number of stacked channel-transformer-blocks
        num_heads (List[int]): Number of heads in `Multi-Head Attention`
        mlp_ratio (float): ratio of hidden layer to input channel in MLP
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        drop_rate (): Dropout ratio of output of Attention. Default: 0.0
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        drop_path_rate (float): Rate required to generate drop path, it will be called by
            `torch.linspace(0, drop_path_rate, depth)`
            designed to increase the probability of DropPath as the depth increases
        act_layer (nn.Module | str): activation function in MLP.
        norm_layer (nn.Module | str): normalization layer before Attention and MLP. Default: None
    """

    def __init__(self,
                 dims,
                 input_resolutions,
                 depths,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop=0.,
                 drop_path_rate=0.1,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 downsamples=None):
        super().__init__()
        self.dims = dims
        self.input_resolutions = input_resolutions
        self.depths = depths
        self.num_heads = num_heads
        self.num_layers = len(dims) // 2

        self.downsamples = cfg_decomposition(downsamples)
        if not self.downsamples:
            self.downsamples = [None] * self.num_layers * 2
        if len(self.downsamples) < self.num_layers * 2:
            self.downsamples.extend([None] * (self.num_layers * 2 - len(self.downsamples)))

        # stochastic depth
        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:len(depths) // 2]))]
        self.dec_dpr = self.dpr[::-1]

        # build layers
        layers = []

        for i in range(self.num_layers * 2):
            dim = self.dims[i]
            input_resolution = self.input_resolutions[i]
            depth = self.depths[i]
            num_heads = self.num_heads[i]
            downsample = self.downsamples[i]

            drop_path = self.dpr[sum(depths[:i]):sum(depths[:i + 1])]
            if i >= self.num_layers:
                drop_path = self.dec_dpr[sum(depths[self.num_layers:i]):sum(depths[self.num_layers:i + 1])]

            # build layer
            layer = ChannelTransformerLayer(dim=dim, input_resolution=input_resolution,
                                            depth=depth, num_heads=num_heads,
                                            mlp_ratio=mlp_ratio,
                                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop_rate=drop_rate, attn_drop=attn_drop,
                                            drop_path=drop_path,
                                            act_layer=act_layer,
                                            norm_layer=norm_layer,
                                            downsample=downsample)

            layers.append(layer)

        self.extend(layers)
