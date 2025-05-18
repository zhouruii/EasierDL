import math
from functools import partial

import torch
from timm.layers import DropPath
from torch import nn
import torch.nn.functional as F

from .common import MLP, GMLP, Linear2d, LayerNorm2d, Permute, SoftmaxSpatial, to_fp32
from .mamba import InitMambaParams
from ..ops.csms6s import selective_scan_fn
from ..ops.triton import cross_scan_fn, cross_merge_fn

from ..builder import MODULE


class SelectiveScan2D(nn.Module):
    """ SS2D in `VMamba`

    scan_mode:
    - 0: Cross2D, Default.
    - 1: single direction, maybe not support.
    - 2: binary direction, maybe not support.
    - 3: Cascade2D, not supported yet.

    Args:
        input_dim ():
        hidden_state ():
        expand_ratio ():
        dt_rank ():
        dt_min ():
        dt_max ():
        dt_init ():
        dt_scale ():
        dt_init_floor ():
        proj_bias (bool): If True, bias will be used in input Linear and output Linea.
        dw_conv_kernel_size ():
        dw_conv_bias ():
        act_layer ():
        drop_rate ():
        channel_first ():
        forward_config ():
        arch_config ():

    """

    def __init__(self,
                 input_dim=256,
                 hidden_state=16,
                 expand_ratio=2,
                 dt_rank='auto',
                 dt_min=0.001,
                 dt_max=0.1,
                 dt_init="random",
                 dt_scale=1.0,
                 dt_init_floor=1e-4,
                 proj_bias=False,
                 dw_conv_kernel_size=3,
                 dw_conv_bias=True,
                 act_layer=nn.SiLU,
                 drop_rate=0.,
                 channel_first=True,
                 forward_config=None,
                 arch_config=None,
                 out_norm_config=None
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_state = hidden_state
        self.expand_ratio = expand_ratio
        self.dt_rank = int(math.ceil(self.input_dim / 16)) if dt_rank == 'auto' else dt_rank
        self.channel_first = channel_first
        self.out_norm_config = out_norm_config

        self.nums_scan = 4
        self.inner_dim = self.input_dim * self.expand_ratio
        # DW Conv
        self.with_dw_conv = dw_conv_kernel_size > 1
        # Linear
        Linear = Linear2d if self.channel_first else nn.Linear

        # get config
        default_arch_cfg = {
            'disable_force32': False,
            'out_act': False,
            'disable_z': True,
            'disable_z_act ': False,
        }
        arch_cfg = default_arch_cfg.copy()
        if arch_config is not None:
            arch_cfg.update(arch_config)
        # set input to fp32
        self.disable_force32 = arch_cfg.get('disable_force32')
        # act layer in output
        self.out_act = arch_cfg.get('out_act')
        # z branch
        self.disable_z = arch_cfg.get('disable_z')
        # act layer in z branch
        self.disable_z_act = arch_cfg.get('disable_z_act')

        # input Linear projection
        input_proj_dim = self.inner_dim if self.disable_z else self.inner_dim * 2
        self.input_proj = Linear(self.input_dim, input_proj_dim, bias=proj_bias)

        # DW Conv, in_channel = out_channel & group = in_channel
        if self.with_dw_conv:
            self.conv2d = nn.Conv2d(
                in_channels=self.inner_dim,
                out_channels=self.inner_dim,
                groups=self.inner_dim,
                kernel_size=dw_conv_kernel_size,
                bias=dw_conv_bias,
                padding=(dw_conv_kernel_size - 1) // 2
            )

        # act layer, default: SiLU
        self.input_act = act_layer()

        # project input to delta,B,C (S_B, S_C, S_delta in `Mamba`)
        # Multiple scanning paths The parameters of each path are independent
        # performed on the SSM branch. The input dimension is still inner dim
        # The dim of delta is controlled by dt_rank. naive: compressed to 1
        self.selective_proj = [
            nn.Linear(self.inner_dim, (self.dt_rank + self.hidden_state * 2), bias=False)
            for _ in range(self.nums_scan)
        ]
        # stack: [param1,...param4] --> (4,param)
        self.selective_proj_weight = nn.Parameter(torch.stack([proj.weight for proj in self.selective_proj], dim=0))
        del self.selective_proj

        # forward core, config in forward_config
        default_fwd_cfg = {
            'force_input_fp32': False,
            'force_output_fp32': True,
            'no_einsum': True,
            'selective_scan_backend': "oflex",
            'scan_mode': 0,
            'scan_force_torch': False
        }
        fwd_cfg = default_fwd_cfg.copy()
        if forward_config is not None:
            fwd_cfg.update(forward_config)
        self.forward_ss2d = partial(self.forward_core, **fwd_cfg)

        # norm layer in SSM output
        self.out_norm = self.get_out_norm(out_norm_config, self.inner_dim, channel_first)

        # output Linear projection
        self.output_act = nn.GELU if self.out_act else nn.Identity()
        self.output_proj = Linear(self.inner_dim, self.input_dim, bias=proj_bias)
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0. else nn.Identity()

        # initialize params: A, D, delta
        # A_log.shape: (K*D,hidden_state), Ds.shape: (K*D,), C_in = dt_rank, C_out = D
        # dts_proj_weight.shape: (K, C_out, C_in), dts_proj_bias.shape: (K, C_out)
        self.A_logs, self.Ds, self.dts_proj_weight, self.dts_proj_bias = InitMambaParams.init_dt_a_d(
            d_state=hidden_state,
            dt_rank=self.dt_rank,
            d_inner=self.inner_dim,
            dt_scale=dt_scale,
            dt_init=dt_init,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init_floor=dt_init_floor,
            k_group=self.nums_scan
        )

    def get_out_norm(self, out_norm_config, inner_dim=192, channel_first=True):
        if out_norm_config is not None:
            out_norm_none = out_norm_config.get('out_norm_none', False)
            out_norm_cnorm = out_norm_config.get('out_norm_cnorm', False)
            out_norm_dw_conv3 = out_norm_config.get('out_norm_dw_conv3', False)
            out_norm_softmax = out_norm_config.get('out_norm_softmax', False)
            out_norm_sigmoid = out_norm_config.get('out_norm_sigmoid', False)
        else:
            out_norm_none, out_norm_cnorm, out_norm_dw_conv3, out_norm_softmax, out_norm_sigmoid \
                = [False, False, False, False, False]

        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm

        if out_norm_none:
            out_norm = nn.Identity()
        elif out_norm_cnorm:
            out_norm = nn.Sequential(
                LayerNorm(inner_dim),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(inner_dim, inner_dim, kernel_size=3, padding=1, groups=inner_dim, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_dw_conv3:
            out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(inner_dim, inner_dim, kernel_size=3, padding=1, groups=inner_dim, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            out_norm = nn.Sigmoid()
        else:
            out_norm = LayerNorm(inner_dim)

        return out_norm

    def forward_core(self,
                     x,
                     force_input_fp32=False,
                     force_output_fp32=True,
                     no_einsum=False,
                     selective_scan_backend=None,
                     scan_mode='Cross2D',
                     scan_force_torch=False
                     ):
        """ core forward function

        Args:
            x (torch.Tensor): Input tensor, with shape: (B,D,H,W)
            force_input_fp32 (bool): If True: input tensor is forced to fp32
            force_output_fp32 (bool): If True: input 16 or 32, then output 32 False: output dtype as input
            no_einsum (bool): Replace einsum with linear or conv1d to raise throughput
            selective_scan_backend (str): Which backend to use for selective scan
            scan_mode (int): Cross2d=0, single direction=1, binary directional=2, cascade2d=3
            scan_force_torch ():

        Returns:

        """
        assert selective_scan_backend in [None, 'torch', 'mamba', 'oflex']
        assert isinstance(scan_mode, int)

        delta_softplus = True
        channel_first = self.channel_first
        out_norm = self.out_norm

        B, D, H, W = x.shape
        K, N, R = self.nums_scan, self.hidden_state, self.dt_rank
        L = H * W

        if scan_mode == 3:
            raise NotImplementedError('Cascade2D is not supported yet!')
        else:
            selective_proj_bias = getattr(self, 'selective_proj_bias', None)
            # multiple scan path, xs.shape:(B,K,D,L)
            # D: inner dim, L: H * W, K: nums_scan
            xs = cross_scan_fn(
                x,
                in_channel_first=True,
                out_channel_first=True,
                scans=scan_mode,
                force_torch=scan_force_torch
            )
            # einsum or naive conv1d
            if no_einsum:
                # selective_proj_weight.shape: (K, dt_rank+hidden_state*2, D)
                # (B, K*D, L) conv with (K*C_out, D), C_out = dt_rank+hidden_state*2
                # x_dt_b_c.shape: (B,K,C_out,L)
                x_dt_b_c = F.conv1d(
                    xs.view(B, -1, L),
                    self.selective_proj_weight.view(-1, D, 1),  # kernel_size=1
                    bias=selective_proj_bias if selective_proj_bias is not None else None,
                    groups=K
                ).view(B, K, -1, L)
                # get deltas, Bs, Cs (s:multiple scan path)
                # dts.shape: (B,K,dt_rank,L), B/C.shape: (B,K,hidden_state,L)
                dts, Bs, Cs = torch.split(x_dt_b_c, [R, N, N], dim=2)
                # project delta from dt_rank to inner dim (broadcast in `Mamba`)
                # dts.shape: (B, K*D, L)
                if hasattr(self, "dts_proj_weight"):
                    dts = F.conv1d(dts.contiguous().view(B, -1, L),
                                   self.dts_proj_weight.view(K * D, -1, 1),
                                   groups=K
                                   )
            else:
                raise NotImplementedError('einsum is not supported yet!')

            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -self.A_logs.to(torch.float).exp()  # (K * D, N)
            Ds = self.Ds.to(torch.float)  # (K * D,)
            Bs = Bs.contiguous().view(B, K, N, L)
            Cs = Cs.contiguous().view(B, K, N, L)
            delta_bias = self.dts_proj_bias.view(-1).to(torch.float)

            if force_input_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            # selective scan core.
            # input shape: (B, K*D, L) same as output shape
            ys = selective_scan_fn(xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus,
                                   oflex=force_output_fp32, backend=selective_scan_backend)
            # (B, K*D, L) --> (B, K, D, H, W)
            ys = ys.view(B, K, -1, H, W)

            # merge multiple scan paths together, y.shape: (B, D, L)
            y = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, force_torch=scan_force_torch)

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)  # (B, L, C)
        y = out_norm(y)

        return y.to(x.dtype)

    def forward(self, x):
        # (B, C, H, W) --> (B, D, H, W)
        x = self.input_proj(x)

        z = None
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # (B, H, W, D)
            if not self.disable_z_act:
                z = self.input_act(z)

        if not self.channel_first:
            x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, D)

        if self.with_dw_conv:
            x = self.conv2d(x)  # (B, D, H, W)

        x = self.input_act(x)

        # y.shape: (B, D, H, W)
        y = self.forward_ss2d(x)

        y = self.output_act(y)

        y = y * z if not self.disable_z else y

        # out.shape: (B, D, H, W)
        out = self.dropout(self.output_proj(y))

        return out


@MODULE.register_module()
class VisualMambaBlock(nn.Module):
    def __init__(self,
                 # --------------- overall --------------- #
                 input_dim=96,
                 drop_path=0.,
                 norm_layer=LayerNorm2d,
                 channel_first=False,
                 post_norm=False,
                 # --------------- SS2D --------------- #
                 hidden_state=16,
                 expand_ratio=2,  # E in Mamba
                 dt_rank='auto',
                 dw_conv_kernel_size=3,
                 dw_conv_bias=True,
                 act_layer=nn.SiLU,
                 drop_rate=0.,
                 forward_config=None,
                 arch_config=None,
                 out_norm_config=None,
                 # --------------- FFN --------------- #
                 gmlp=False,
                 mlp_ratio=4,
                 mlp_act_layer=nn.GELU,
                 mlp_drop_rate=0.,
                 ):
        super().__init__()
        self.ssm_branch = hidden_state > 0
        self.mlp_branch = mlp_ratio > 0
        self.post_norm = post_norm

        # SSM Block
        if self.ssm_branch:
            self.first_norm = norm_layer(input_dim)
            self.ssm = SelectiveScan2D(
                input_dim=input_dim,
                hidden_state=hidden_state,
                expand_ratio=expand_ratio,
                dt_rank=dt_rank,
                dt_min=0.001,
                dt_max=0.1,
                dt_init="random",
                dt_scale=1.0,
                dt_init_floor=1e-4,
                proj_bias=False,
                dw_conv_kernel_size=dw_conv_kernel_size,
                dw_conv_bias=dw_conv_bias,
                act_layer=act_layer,
                drop_rate=drop_rate,
                channel_first=channel_first,
                forward_config=forward_config,
                arch_config=arch_config,
                out_norm_config=out_norm_config
            )

        self.drop_path = DropPath(drop_path)

        # FFN branch
        if self.mlp_branch:
            _MLP = MLP if not gmlp else GMLP
            self.second_norm = norm_layer(input_dim)
            mlp_hidden_dim = int(mlp_ratio * input_dim)
            self.FFN = _MLP(
                in_features=input_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=mlp_act_layer,
                drop=mlp_drop_rate,
                channel_first=channel_first
            )

    def forward(self, x):
        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.first_norm(self.ssm(x)))
            else:
                x = x + self.drop_path(self.ssm(self.first_norm(x)))

        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.second_norm(self.FFN(x)))
            else:
                x = x + self.drop_path(self.FFN(self.second_norm(x)))

        return x
