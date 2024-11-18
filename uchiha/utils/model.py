import math
import warnings

import torch
from torch import nn


def count_parameters(model):
    """ Calculate the parameters of the model.

    Args:
        model (nn.Module): The model that require counting parameter quantities

    Returns:
        int: Parameter quantities of the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
    elif isinstance(act, nn.Module):
        return act
    else:
        warnings.warn(f'activation function:{act} is not supported yet! '
                      f'this string will be used directly. ')


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


def check_postfix(tag, value):
    ret = value[-len(tag):] == tag
    if ret:
        value = value[:-len(tag)]
    return ret, value


def to_fp32(*args):
    return (_a.to(torch.float32) for _a in args)
