from thop import profile, clever_format
import torch

from uchiha.models.builder import build_model
from uchiha.utils import load_config


def flops():
    cfg = load_config('configs/hdr_former/AVIRIS/ours/baseline_v4.yaml')

    # model
    model = build_model(cfg.model.to_dict())
    model.eval()

    B = 1
    C = 224
    H = 128
    W = 128

    example_input = torch.randn(B, C, H, W)
    macs, params = profile(model, inputs=(example_input,))
    macs, params = clever_format([macs, params], "%.3f")

    print(f"FLOPs: {macs}")
    print(f"Parameters: {params}")


if __name__ == '__main__':
    flops()
