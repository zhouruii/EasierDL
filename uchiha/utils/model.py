import math
import warnings

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


def build_norm(norm_layer):
    """ Build a Normalization Layer

    Args:
        norm_layer (str): The type of Normalization Layer

    Returns:
        nn.Module: The built Normalization Layer
    """
    if norm_layer == 'nn.LayerNorm':
        return nn.LayerNorm
    else:
        warnings.warn(f'norm_layer:{norm_layer} is not supported yet! '
                      f'this string will be used directly. ')