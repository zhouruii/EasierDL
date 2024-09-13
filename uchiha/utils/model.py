import math


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sequence_to_image(x):
    # x.shape: B, L, C
    B, L, C = x.shape
    H, W = int(math.sqrt(L)), int(math.sqrt(L))
    return x.view(B, H, W, C).permute(0, 3, 1, 2)
