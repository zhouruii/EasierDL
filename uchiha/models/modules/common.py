import math
import numbers
import warnings

import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F
from torch.nn import init


def initialize_weights(module):
    """智能权重初始化（自动识别层类型）"""
    if isinstance(module, nn.Conv2d):
        # 卷积层：Kaiming初始化（适配ReLU）
        init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            init.constant_(module.bias, 0)

    elif isinstance(module, nn.Linear):
        # 全连接层：Xavier初始化
        init.xavier_normal_(module.weight)
        if module.bias is not None:
            init.constant_(module.bias, 0)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class MLP(nn.Module):
    """ MLP in Channel Transformer

    Args:
        in_features (int): number of input channels
        hidden_features (int): number of hidden layer channels
        out_features (int): number of output channels
        act_layer (nn.Module): activation function
        drop (float): the rate of `Dropout` layer. Default: 0.0
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.,
                 channel_first=False
                 ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channel_first else nn.Linear

        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class GMLP(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.,
                 channel_first=False):
        super().__init__()
        self.channel_first = channel_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channel_first else nn.Linear

        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x


class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                             error_msgs)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class SoftmaxSpatial(nn.Softmax):
    def forward(self, x: torch.Tensor):
        if self.dim == -1:
            B, C, H, W = x.shape
            return super().forward(x.view(B, C, -1)).view(B, C, H, W)
        elif self.dim == 1:
            B, H, W, C = x.shape
            return super().forward(x.view(B, -1, C)).view(B, H, W, C)
        else:
            raise NotImplementedError


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFreeLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        _, C, H, W = x.shape
        x = to_3d(x)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        out = x / torch.sqrt(sigma + 1e-5) * self.weight
        return to_4d(out, H, W)


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBiasLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        _, C, H, W = x.shape
        x = to_3d(x)
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        out = (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
        return to_4d(out, H, W)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """ 3x3 Convolution Layer

    Args:
        in_planes (int): number of input channels
        out_planes (int): number of output channels
        stride (int): the stride for the convolution operation. Default: 1
        groups (int): the groups for the convolution operation. Default: 1
        dilation (int): the dilation for the convolution operation. Default: 1

    Returns:
        nn.Conv2d: Convolution Layer
    """

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """ 1x1 Convolution Layer

    Args:
        in_planes (int): number of input channels
        out_planes (int): number of output channels
        stride (int): the stride for the convolution operation. Default: 1

    Returns:
        nn.Conv2d: Convolution Layer
    """

    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def to_fp32(*args):
    return (_a.to(torch.float32) for _a in args)


def build_norm(norm_layer):
    """ Build a Normalization Layer

    Args:
        norm_layer (str | nn.Module): The type of Normalization Layer

    Returns:
        nn.Module: The built Normalization Layer
    """
    if norm_layer == 'nn.LayerNorm':
        return nn.LayerNorm
    elif isinstance(norm_layer, nn.Module):
        return norm_layer
    else:
        warnings.warn(f'norm_layer:{norm_layer} is not supported yet! '
                      f'this string will be used directly. ')


def build_act(act):
    """ Build activation function

    Args:
        act (str): The type of activation function

    Returns:
        nn.Module: The built Normalization Layer
    """
    if act == 'nn.GELU':
        return nn.GELU
    elif act == 'nn.ReLU':
        return nn.ReLU
    elif act == 'nn.Sigmoid':
        return nn.Sigmoid
    elif isinstance(act, nn.Module):
        return act
    else:
        warnings.warn(f'activation function:{act} is not supported yet! '
                      f'this string will be used directly. ')


def cfg_decomposition(cfg):
    """ Decompose the config to list of config

    The value of input config information (dict) is a list,
    decompose elements in list, each build a new config information,
    and compose these new config in a list.

    Args:
        cfg (List[dict] | dict): Configuration information, where the first key is type,
            values of input config are list

    Returns:
        list: A list containing decomposed config
    """
    if isinstance(cfg, list):
        return cfg

    if cfg is None:
        return

    def helper(_cfg):
        new_cfg = {}
        for key, value in _cfg.items():
            if isinstance(value, list):
                new_cfg[key] = value.pop(0)
                if len(value) == 0:
                    _cfg[key] = None
            elif isinstance(value, int):
                new_cfg[key] = value
                _cfg[key] = None
            elif value is None:
                return
            else:
                new_cfg[key] = value
        return new_cfg

    decomposed = []
    while True:
        decomposed_cfg = helper(cfg)
        if decomposed_cfg:
            decomposed.append(decomposed_cfg)
        else:
            break
    return decomposed


def sequence_to_image(x):
    """ (B, L, C) --> (B, C, H, W)

    Args:
        x (Tensor): Sequential data (B, L, C).

    Returns:
        Tensor: Image data (B, C, H, W).
    """
    # x.shape: B, L, C
    B, L, C = x.shape
    H, W = int(math.sqrt(L)), int(math.sqrt(L))
    return x.view(B, H, W, C).permute(0, 3, 1, 2)


class SingleBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=1.0, eps=1e-5):
        super(SingleBatchNorm, self).__init__(
            num_features,
            affine=True,
            track_running_stats=True,
            momentum=momentum,
            eps=eps
        )
        self.running_var = None
        self.running_mean = None

    def forward(self, x: torch.Tensor):
        # 如果是训练模式，只使用 running stats
        if self.training:
            # 强制使用 running mean/var
            exponential_average_factor = self.momentum

            # 手动调用 _check_input_dim
            self._check_input_dim(x)

            # 计算当前 batch 的 mean & var（我们不使用）
            mean = x.mean([0, 2, 3])
            var = x.var([0, 2, 3], unbiased=False)

            # 更新 running mean/var（可选）
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean + (
                        1 - exponential_average_factor) * self.running_mean
                self.running_var = exponential_average_factor * var + (
                        1 - exponential_average_factor) * self.running_var

            # 实际归一化时仍使用 running stats
            mean = self.running_mean
            var = self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        x = (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
        if self.affine:
            x = x * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return x


class GatedUnit(nn.Module):
    def __init__(self, in_channels, depth=1, kernel_size=3, stride=1, padding=1):
        super(GatedUnit, self).__init__()
        # 门控分支，卷积 + sigmoid
        self.gate_conv = nn.ModuleList(nn.Conv2d(in_channels, in_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding,
                                                 groups=in_channels) for _ in range(depth))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gate = x
        for m in self.gate_conv:
            gate = m(gate)
        # 门控分支
        gate = self.sigmoid(gate)  # [B, C, H, W]
        # 门控输入
        out = x * gate
        return out
