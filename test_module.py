import traceback

import pytest
import torch
from uchiha.models.components.preprocessor import GroupRCP
from uchiha.models.modules.cross_transformer import SelfCrossAttention


def template(params, module, inp_data):
    instance_module = module(**params)
    # forward
    try:
        out = instance_module(inp_data)
        print(f"\nForward pass succeeded! Output shape: {out.shape}")
    except Exception as e:
        pytest.fail(f"\nForward pass failed：{str(e)}\n{traceback.format_exc()}")


def test_group_rcp():
    # 模块参数
    params = {
        "split_bands": [100],
    }

    # 测试数据
    x = torch.randn(2, 128, 512, 512)  # [batch, channel, height, width]

    # 测试
    template(params, GroupRCP, x)


def test_self_cross_attention():
    # 模块参数
    params = {
        "in_channels": 128,
        "num_heads": 8,
        "extra_v_branch": False,
        "freq_cfg": {
            "type": 'DWT',
            "in_channels": 128,
            "J": 1,
            "wave": 'haar',
            "mode": 'reflect'
        },
    }

    # 测试数据
    x = torch.randn(2, 128, 512, 512)  # [batch, channel, height, width]

    # 测试
    template(params, SelfCrossAttention, x)
