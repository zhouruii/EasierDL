import os
import random

import cv2
from tqdm import tqdm

from gen_streak import generate_bird_view_streak
from util import downsample_image


def generate_small_rain(num_sets, output_dir='/home/disk2/ZR/datasets/OurHSI/streakV2'):
    """
    批量生成雨纹图片。

    参数:
    num_sets (int): 需要生成的雨纹组数。
    """
    # 创建保存图片的文件夹
    os.makedirs(output_dir, exist_ok=True)

    os.makedirs(os.path.join(output_dir, "small"), exist_ok=True)

    for i in tqdm(range(num_sets), desc='synthesizing small rain...'):
        params = {"height": 1024, "width": 1024, "depth": 319, "num_drops": random.randint(1900, 2000),
                  "streak_length": random.randint(25, 30), "wind_angle": random.randint(-180, 180),
                  "wind_strength": random.uniform(0, 0.05), "f": 512}
        # 生成雨纹
        streak = generate_bird_view_streak(**params)
        # 下采样
        down = downsample_image(streak, scale_factor=1 / 4)
        max_val = down.max()
        cv2.normalize(down, down, 0, max_val * 3, cv2.NORM_MINMAX)
        cv2.imwrite(os.path.join(output_dir, "small", f"{i + 1}.jpg"), down)


if __name__ == '__main__':
    generate_small_rain(6000)
