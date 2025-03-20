import os

import cv2

from gen_streak import generate_bird_view_streak, smooth_image
from config import RAIN_STREAK_BATCH, RAIN_STREAK_BATCH_V2
from util import normalize


def downsample_image(image, scale_factor):
    """
    对图像进行下采样。

    参数:
    image (numpy.ndarray): 输入图像，形状为 (height, width, channels)。
    scale_factor (float): 下采样比例，小于1的值表示缩小图像。

    返回:
    numpy.ndarray: 下采样后的图像。
    """
    if scale_factor > 1:
        raise ValueError("scale_factor必须小于1以进行下采样。")

    new_height = int(image.shape[0] * scale_factor)
    new_width = int(image.shape[1] * scale_factor)

    downsampled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    return smooth_image(downsampled_image)


def generate_small_rain(num_sets, output_dir='streak/OurHSI'):
    """
    批量生成雨纹图片。

    参数:
    num_sets (int): 需要生成的雨纹组数。
    """
    # 创建保存图片的文件夹
    os.makedirs(output_dir, exist_ok=True)

    os.makedirs(os.path.join(output_dir, "small"), exist_ok=True)

    for i in range(num_sets):
        # 生成雨纹
        streak = generate_bird_view_streak(**RAIN_STREAK_BATCH["small"])
        # 下采样
        down = downsample_image(streak, scale_factor=1 / 4)
        max_val = down.max()
        cv2.normalize(down, down, 0, max_val * 3, cv2.NORM_MINMAX)
        # 保存原始和下采样后的图片
        # cv2.imwrite(os.path.join(raw_dir, "medium", f"{i + 1}.jpg"), streak)
        cv2.imwrite(os.path.join(output_dir, "small", f"{i + 1}.jpg"), down)


def generate_small_rain_v2(num_sets, output_dir='streak/AVIRIS'):
    """
    批量生成雨纹图片。

    参数:
    num_sets (int): 需要生成的雨纹组数。
    """
    # 创建保存图片的文件夹
    os.makedirs(output_dir, exist_ok=True)

    os.makedirs(os.path.join(output_dir, "small"), exist_ok=True)

    for i in range(num_sets):
        # 生成雨纹
        streak = generate_bird_view_streak(**RAIN_STREAK_BATCH_V2["small"])
        # 下采样
        down = downsample_image(streak, scale_factor=1 / 1)
        max_val = down.max()
        # cv2.normalize(down, down, 0, max_val * 1.5, cv2.NORM_MINMAX)
        # 保存原始和下采样后的图片
        # cv2.imwrite(os.path.join(raw_dir, "medium", f"{i + 1}.jpg"), streak)
        cv2.imwrite(os.path.join(output_dir, "small", f"{i + 1}.jpg"), down)


if __name__ == '__main__':
    generate_small_rain_v2(10)
