import math
import warnings

from torch import nn


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sequence_to_image(x):
    # x.shape: B, L, C
    B, L, C = x.shape
    H, W = int(math.sqrt(L)), int(math.sqrt(L))
    return x.view(B, H, W, C).permute(0, 3, 1, 2)


def build_norm(norm_layer):
    if norm_layer == 'nn.LayerNorm':
        return nn.LayerNorm
    else:
        warnings.warn(f'norm_layer:{norm_layer} is not supported yet! '
                      f'this string will be used directly. ')