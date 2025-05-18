# refer to `Mamba`, `Mamba-2` and `VMamba`
import math

import torch
from torch import nn
from einops import rearrange

try:
    from mamba_ssm import Mamba2, Mamba
except ImportError:
    Mamba2, Mamba = None, None

# ------------------------------- Mamba-1 ------------------------------- #
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None
# ------------------------------- causal conv1d ------------------------------- #
try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

from ..builder import MODULE

MODULE.register_module(module=Mamba2, name='Mamba2Block')
MODULE.register_module(module=Mamba, name='MambaBlock')


# TODO 将VMamba中其他简单的初始化方法集成进来
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
            nn.Linear
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
    def a_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
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
    def d_init(d_inner, copies=-1, device=None, merge=True):
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
    def init_dt_a_d(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
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
        # init A and D
        A_logs = cls.a_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
        Ds = cls.d_init(d_inner, copies=k_group, merge=True)  # (K * D)

        # dt_projections, project delta from dt_rank to D dim
        if k_group > 0:
            dt_projections = [
                cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
                for _ in range(k_group)
            ]
            dt_projections_weight = nn.Parameter(
                torch.stack([t.weight for t in dt_projections], dim=0))  # (K, inner, rank)
            dt_projections_bias = nn.Parameter(torch.stack([t.bias for t in dt_projections], dim=0))  # (K, inner)
            del dt_projections

            return A_logs, Ds, dt_projections_weight, dt_projections_bias
        else:
            dt_projections = cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            return A_logs, Ds, dt_projections


@MODULE.register_module()
class ClassicMambaBlock(nn.Module):
    def __init__(self,
                 input_dim=256,
                 hidden_state=16,
                 expand_ratio=2,
                 dt_rank="auto",
                 dt_min=0.001,
                 dt_max=0.1,
                 dt_init="random",
                 dt_scale=1.0,
                 dt_init_floor=1e-4,
                 dw_conv_bias=True,
                 dw_conv_kernel_size=3,
                 proj_bias=False,
                 use_fast_path=False,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_state = hidden_state
        self.expand_ratio = expand_ratio
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor
        self.use_fast_path = use_fast_path

        self.inner_dim = self.input_dim * self.expand_ratio
        self.dt_rank = math.ceil(self.input_dim / 16) if dt_rank == "auto" else dt_rank

        # input projection: x --> X and Z
        self.input_proj = nn.Linear(self.input_dim, self.inner_dim * 2, bias=proj_bias)

        # conv following Linear projection
        self.conv1d = nn.Conv1d(
            in_channels=self.inner_dim,
            out_channels=self.inner_dim,
            kernel_size=dw_conv_kernel_size,
            bias=dw_conv_bias,
            groups=self.inner_dim,
            padding=dw_conv_kernel_size - 1  # for causal mode
        )

        # use after conv1d
        self.act = nn.SiLU()

        # X --> dt, B, C
        self.selective_proj = nn.Linear(
            self.inner_dim,
            self.dt_rank + 2 * self.hidden_state,
            bias=False
        )

        # dt projection, from dt_rank to inner dim
        # dt, A, D initialization
        self.A_log, self.D, self.dt_proj = self.init_params()

        # output projection, from inner dim to input dim
        self.output_proj = nn.Linear(
            in_features=self.inner_dim,
            out_features=self.input_dim,
            bias=proj_bias
        )

    def init_params(self):
        return InitMambaParams.init_dt_a_d(
            d_state=self.hidden_state,
            d_inner=self.inner_dim,
            dt_rank=self.dt_rank,
            dt_scale=self.dt_scale,
            dt_init=self.dt_init,
            dt_min=self.dt_min,
            dt_max=self.dt_max,
            dt_init_floor=self.dt_init_floor,
            k_group=-1
        )

    def forward(self, x):
        # x.shape: (B, L, C)
        L = x.shape[1]
        # input projection: (B, L, C) --> (B, L, 2 * E * C)
        xz = self.input_proj(x)
        A = -torch.exp(self.A_log.float())  # (D, N) ?

        if self.use_fast_path:
            assert causal_conv1d_fn is not None, 'fast_path must use causal conv1d'
            raise NotImplementedError('fast path in Mamba not supported yet!')
        else:
            x, z = xz.transpose(1, 2).chunk(2, dim=1)

            # conv (deep wise)
            if causal_conv1d_fn is not None:
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, 'd 1 k -> d k'),  # 1: deep wise
                    bias=self.conv1d.bias,
                    activation='silu'
                )  # x.shape: (B, D, L)
            else:
                x = self.act(self.conv1d(x)[:, L])

            # selective projection, dt_b_c.shape: (B, dt_rank + 2 * N, L)
            dt_b_c = self.selective_proj(rearrange(x, 'b d l -> b l d')).transpose(1, 2)
            # Order: (B, dt_rank, L), (B, N, L), (B, N, L)
            dt, B, C = torch.split(
                tensor=dt_b_c,
                split_size_or_sections=[self.dt_rank, self.hidden_state, self.hidden_state],
                dim=1
            )

            # dt projection: dt_rank --> inner dim
            dt = rearrange(dt, 'b d l -> d (b l)')
            # (D, dt_rank) @ (dt_rank, B*L)
            dt = self.dt_proj.weight @ dt
            dt = rearrange(dt, 'd (b l) -> b d l', l=L)

            B = B.contiguous()
            C = C.contiguous()

            # selective scan, y.shape: (B, D, L)
            y = selective_scan_fn(
                u=x,
                delta=dt,
                A=A,
                B=B,
                C=C,
                D=self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias,
                delta_softplus=True,
            )

            out = self.output_proj(y.transpose(1, 2))

        return out


@MODULE.register_module()
class ChannelMambaBlock(nn.Module):
    def __init__(self,
                 seq_len=16,
                 hidden_state=16,
                 expand_ratio=2,
                 factor=4.0,
                 dt_rank="auto",
                 dt_min=0.001,
                 dt_max=0.1,
                 dt_init="random",
                 dt_scale=1.0,
                 dt_init_floor=1e-4,
                 dw_conv_bias=True,
                 dw_conv_kernel_size=3,
                 proj_bias=False,
                 use_fast_path=False,
                 ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_state = hidden_state
        self.expand_ratio = expand_ratio
        self.factor = factor
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor
        self.use_fast_path = use_fast_path

        self.inner_dim = int(self.seq_len * self.expand_ratio * self.factor)
        self.dt_rank = math.ceil(self.seq_len / 16) if dt_rank == "auto" else dt_rank

        # spatial projection.
        self.spatial_proj = nn.Linear(self.seq_len, int(self.seq_len * self.factor))

        # input projection: x --> X and Z
        self.input_proj = nn.Linear(int(self.seq_len * self.factor), self.inner_dim * 2, bias=proj_bias)

        # conv following Linear projection
        self.conv1d = nn.Conv1d(
            in_channels=self.inner_dim,
            out_channels=self.inner_dim,
            kernel_size=dw_conv_kernel_size,
            bias=dw_conv_bias,
            groups=self.inner_dim,  # self.inner_dim ?
            padding=dw_conv_kernel_size - 1  # for causal mode
        )

        # use after conv1d
        self.act = nn.SiLU()

        # X --> dt, B, C
        self.selective_proj = nn.Linear(
            self.inner_dim,
            self.dt_rank + 2 * self.hidden_state,
            bias=False
        )

        # dt projection, from dt_rank to inner dim
        # dt, A, D initialization
        self.A_log, self.D, self.dt_proj = self.init_params()

        # output projection, from inner dim to input dim
        self.output_proj = nn.Linear(
            in_features=self.inner_dim,
            out_features=self.seq_len,
            bias=proj_bias
        )

    def init_params(self):
        return InitMambaParams.init_dt_a_d(
            d_state=self.hidden_state,
            d_inner=self.inner_dim,
            dt_rank=self.dt_rank,
            dt_scale=self.dt_scale,
            dt_init=self.dt_init,
            dt_min=self.dt_min,
            dt_max=self.dt_max,
            dt_init_floor=self.dt_init_floor,
            k_group=-1
        )

    def forward(self, x):
        # x.shape: (B, L, C)
        x = rearrange(x, 'b l d -> b d l')
        L = x.shape[1]
        # (B, D, L) --> (B, D, L * f)
        x = self.spatial_proj(x)

        # input projection: (B, L, C) --> (B, L, 2 * E * C)
        xz = self.input_proj(x)
        A = -torch.exp(self.A_log.float())  # (D, N) ?

        if self.use_fast_path:
            assert causal_conv1d_fn is not None, 'fast_path must use causal conv1d'
            raise NotImplementedError('fast path in Mamba not supported yet!')
        else:
            x, z = xz.transpose(1, 2).chunk(2, dim=1)

            # conv (deep wise)
            if causal_conv1d_fn is not None:
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, 'd 1 k -> d k'),  # 1: deep wise
                    bias=self.conv1d.bias,
                    activation='silu'
                )  # x.shape: (B, D, L)
            else:
                x = self.act(self.conv1d(x)[:, L])

            # selective projection, dt_b_c.shape: (B, dt_rank + 2 * N, L)
            dt_b_c = self.selective_proj(rearrange(x, 'b d l -> b l d')).transpose(1, 2)
            # Order: (B, dt_rank, L), (B, N, L), (B, N, L)
            dt, B, C = torch.split(
                tensor=dt_b_c,
                split_size_or_sections=[self.dt_rank, self.hidden_state, self.hidden_state],
                dim=1
            )

            # dt projection: dt_rank --> inner dim
            dt = rearrange(dt, 'b d l -> d (b l)')
            # (D, dt_rank) @ (dt_rank, B*L)
            dt = self.dt_proj.weight @ dt
            dt = rearrange(dt, 'd (b l) -> b d l', l=L)

            B = B.contiguous()
            C = C.contiguous()

            # selective scan, y.shape: (B, D, L)
            y = selective_scan_fn(
                u=x,
                delta=dt,
                A=A,
                B=B,
                C=C,
                D=self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias,
                delta_softplus=True,
            )

            out = self.output_proj(y.transpose(1, 2))

        return out.transpose(1, 2)

@MODULE.register_module()
class ChannelMambaLayer(nn.Module):
    def __init__(self,
                 depth=2,
                 seq_len=16,
                 hidden_state=16,
                 expand_ratio=2,
                 factor=4.0,
                 dt_rank="auto",
                 dt_min=0.001,
                 dt_max=0.1,
                 dt_init="random",
                 dt_scale=1.0,
                 dt_init_floor=1e-4,
                 dw_conv_bias=True,
                 dw_conv_kernel_size=3,
                 proj_bias=False,
                 use_fast_path=False,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([
            ChannelMambaBlock(
                seq_len=seq_len,
                hidden_state=hidden_state,
                expand_ratio=expand_ratio,
                factor=factor,
                dt_rank=dt_rank,
                dt_min=dt_min,
                dt_max=dt_max,
                dt_init=dt_init,
                dt_scale=dt_scale,
                dt_init_floor=dt_init_floor,
                dw_conv_bias=dw_conv_bias,
                dw_conv_kernel_size=dw_conv_kernel_size,
                proj_bias=proj_bias,
                use_fast_path=use_fast_path,
            ) for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
