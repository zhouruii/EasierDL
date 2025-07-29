import argparse

import imageio
import numpy as np

from tools.visualization.show import TwoPercentLinear
from uchiha.apis import set_random_seed
from uchiha.apis.inference_dehaze import single_inference, load_data, normalize
from uchiha.utils import load_config


def parse_args():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--seed', type=int, default=49)
    args_parser.add_argument('--config', type=str, default='configs/hsi_dehaze/HD/inference/D3.yaml')
    return args_parser.parse_args()


def inference():
    args = parse_args()
    cfg = load_config(args.config)
    # random seed
    set_random_seed(args.seed)
    single_inference(cfg)


if __name__ == '__main__':
    inference()

    # data = load_data('/media/mango/系统/6636数据/2/mk/dataset/HD/train/haze/1_15.tif')
    # data = normalize(data)
    # data = np.stack([
    #     np.clip(data[:, :, 58], 0, 1),  # R
    #     np.clip(data[:, :, 37], 0, 1),  # G
    #     np.clip(data[:, :, 19], 0, 1)  # B
    # ], axis=-1)
    #
    # data = np.uint8(data * 255)
    # data = TwoPercentLinear(data[:, :, :])  # (2, 1, 0)
    #
    # imageio.imwrite('org.png', data)
