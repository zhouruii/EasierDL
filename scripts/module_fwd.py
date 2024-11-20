import time

import torch
from einops import rearrange

from mamba_ssm import Mamba2
from torch import nn

from causal_conv1d import causal_conv1d_fn
import causal_conv1d_cuda

from uchiha import ClassicMambaBlock
from uchiha.models.basemodules.mamba import ChannelMambaBlock
from uchiha.models.basemodules.visual_mamba import SelectiveScan2D, VisualMambaBlock


def bench():
    # x = torch.randn(1, 96, 56, 56).cuda()
    x = torch.randn(2, 9, 256).cuda()
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

    Mamba2Block = Mamba2(
        d_model=256,  # Model dimension d_model
        d_state=128,  # SSM state expansion factor, typically 64 or 128
        d_conv=4,  # Local convolution width
        expand=2,  # Block expansion factor
        headdim=64,
    ).cuda()

    ClassicMamba = ClassicMambaBlock(
        input_dim=256,
        hidden_state=16,
        expand_ratio=2,
    ).cuda()

    ChannelMamba = ChannelMambaBlock(
        seq_len=9,
        hidden_state=16,
        expand_ratio=2,
    ).cuda()

    print(x.shape)

    tim = time.time()
    y = ChannelMamba(x)
    print(time.time() - tim)

    print(y.shape)


def module_demo():
    x = torch.randn(2, 9, 768).cuda()

    module = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=4, padding=3, groups=768).cuda()
    weight = rearrange(module.weight, 'c 1 k -> c k')
    y = causal_conv1d_cuda.causal_conv1d_fwd(x, weight)

    return y


if __name__ == '__main__':
    bench()

    # module_demo()
