import os
import warnings

import numpy as np


def check_zero_channel(data):
    """检查 HWC 数据是否存在某个通道均为 0 的情况。

    Args:
        data (np.ndarray): 形状为 (H, W, C) 的数据。

    Returns:
        list: 均为 0 的通道索引列表。
    """
    # 确保数据是 HWC 格式
    if data.ndim != 3:
        raise ValueError("数据维度不是 HWC 格式")

    # 检查每个通道是否均为 0
    zero_channels = [c for c in range(data.shape[2]) if np.all(data[:, :, c] == 0)]
    return zero_channels


def process_directory(directory):
    """处理目录下的所有 .npy 文件，检查是否存在某个通道均为 0 的情况。

    Args:
        directory (str): 目录路径。
    """
    # 遍历目录下的所有文件
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            file_path = os.path.join(directory, filename)
            print(f"正在处理文件: {file_path}")

            # 加载 .npy 文件
            data = np.load(file_path)

            # 检查是否存在某个通道均为 0
            zero_channels = check_zero_channel(data)
            if zero_channels:
                warnings.warn(f"文件 {filename} 中以下通道均为 0: {zero_channels}")
            else:
                print(f"文件 {filename} 中没有通道均为 0")


# 示例使用
if __name__ == "__main__":
    # 目录路径
    directory = "/home/disk2/ZR/datasets/AVIRIS/512/gt"  # 替换为你的目录路径

    # 处理目录下的所有 .npy 文件
    process_directory(directory)
