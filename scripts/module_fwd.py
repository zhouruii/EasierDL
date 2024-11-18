import time

import torch

from uchiha.models.basemodules.visual_mamba import SelectiveScan2D, VisualMambaBlock


def bench(module, x):
    print(x.shape)

    tim = time.time()
    y = module(x)
    print(time.time() - tim)

    print(y.shape)


if __name__ == '__main__':
    x = torch.randn(1, 96, 56, 56).cuda()
    SS2D = SelectiveScan2D(
        input_dim=96,
        hidden_state=16,
        expand_ratio=2,
        channel_first=True
    ).cuda()

    VSSBlock = VisualMambaBlock(
        input_dim=96,
        hidden_state=16,
        expand_ratio=2,
        channel_first=True
    ).cuda()

    bench(VSSBlock, x)

