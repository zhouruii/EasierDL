import torch

try:
    from mamba_ssm import Mamba2, Mamba
except:
    Mamba2, Mamba = None, None

from ..builder import BASEMODULE

BASEMODULE.register_module(module=Mamba2, name='Mamba2Block')
BASEMODULE.register_module(module=Mamba, name='MambaBlock')

if __name__ == '__main__':

    batch, length, dim = 2, 64, 256
    x = torch.randn(batch, length, dim).to("cuda")
    model = Mamba2(
        # This module uses roughly 3 * expand * d_model^2 parameters
        d_model=dim,  # Model dimension d_model
        d_state=64,  # SSM state expansion factor, typically 64 or 128
        d_conv=4,  # Local convolution width
        expand=2,  # Block expansion factor
    ).to("cuda")
    y = model(x)

    # model = Mamba(
    #     # This module uses roughly 3 * expand * d_model^2 parameters
    #     d_model=dim,  # Model dimension d_model
    #     d_state=16,  # SSM state expansion factor
    #     d_conv=4,  # Local convolution width
    #     expand=2,  # Block expansion factor
    # ).to("cuda")
    # y = model(x)
    assert y.shape == x.shape