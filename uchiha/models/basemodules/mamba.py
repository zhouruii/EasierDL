import math

import torch
from torch import nn

try:
    from mamba_ssm import Mamba2, Mamba
except:
    Mamba2, Mamba = None, None

from ..builder import BASEMODULE

BASEMODULE.register_module(module=Mamba2, name='Mamba2Block')
BASEMODULE.register_module(module=Mamba, name='MambaBlock')


# TODO 将VMamba其他的初始化方法集成进来
class InitMambaParams:

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        """ init delta

        ((B), (L), D) in Mamba

        Args:
            dt_rank ():
            d_inner ():
            dt_scale ():
            dt_init ():
            dt_min ():
            dt_max ():
            dt_init_floor ():

        Returns:

        """
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        """ init A

        (D,N) in Mamba, (K,D,N) in VMamba (D --> inner dim)

        Args:
            d_state ():
            d_inner ():
            copies ():
            device ():
            merge ():

        Returns:

        """
        # S4D real initialization
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        # VMamba will use it due to multiple scans
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        """ init D

        (D) in Mamba, (K,D) in VMamba (D --> inner dim)

        Args:
            d_inner ():
            copies ():
            device ():
            merge ():

        Returns:

        """
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        """ init params: delta, A, D

        used in VMamba

        Args:
            d_state ():
            dt_rank ():
            d_inner ():
            dt_scale ():
            dt_init ():
            dt_min ():
            dt_max ():
            dt_init_floor ():
            k_group ():

        Returns:

        """
        # dt_projections, project delta from dt_rank to D dim
        dt_projections = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        dt_projections_weight = nn.Parameter(torch.stack([t.weight for t in dt_projections], dim=0))  # (K, inner, rank)
        dt_projections_bias = nn.Parameter(torch.stack([t.bias for t in dt_projections], dim=0))  # (K, inner)
        del dt_projections

        # init A and D
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True)  # (K * D)
        return A_logs, Ds, dt_projections_weight, dt_projections_bias


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
