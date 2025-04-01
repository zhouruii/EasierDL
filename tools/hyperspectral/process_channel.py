import os
import numpy as np
from scipy.io import loadmat

from utils import normalize


def remove_channels_from_npy(file_path, channels_to_remove):
    """从 .npy 文件中删除指定通道的数据。

    Args:
        file_path (str): .npy 文件路径。
        channels_to_remove (list): 需要删除的通道索引列表。

    Returns:
        np.ndarray: 删除指定通道后的数据。
    """
    # 加载 .npy 文件
    if file_path.endswith('.mat'):
        data = loadmat(file_path)['hsi_0']
    else:
        data = np.load(file_path)

    # 确保数据是 HWC 格式（高度、宽度、通道）
    if data.ndim != 3:
        raise ValueError(f"文件 {file_path} 的维度不是 HWC 格式")

    # 获取需要保留的通道索引
    channels_to_keep = [i for i in range(data.shape[2]) if i not in channels_to_remove]

    # 删除指定通道
    result = data[:, :, channels_to_keep]

    return normalize(result)


def process_directory(directory, channels_to_remove, output_dir=None):
    """处理目录下的所有 .npy 文件，删除指定通道并保存。

    Args:
        directory (str): 目录路径。
        channels_to_remove (list): 需要删除的通道索引列表。
    """

    if output_dir is None:
        output_dir = directory
    else:
        os.makedirs(output_dir, exist_ok=True)

    # 遍历目录下的所有文件
    filenames = os.listdir(directory)
    for idx, filename in enumerate(filenames):
        if filename.endswith(".npy") or filename.endswith(".mat"):
            file_path = os.path.join(directory, filename)
            # 删除指定通道
            result = remove_channels_from_npy(file_path, channels_to_remove)

            if filename.endswith(".mat"):
                filename = filename.replace('.mat', '.npy')
            save_path = os.path.join(output_dir, filename)

            # 保存修改后的数据（文件名不变）
            np.save(save_path, result)
            print(f"Processed: [{idx}/{len(filenames)}], File: {file_path}, Remove: {channels_to_remove}, "
                  f"saved to --> {save_path}")


# 示例使用
if __name__ == "__main__":
    data = np.load('/home/disk2/ZR/datasets/OurHSI/extra/gt/1_11_1_1.npy')
    # 目录路径
    directory = "/home/disk2/ZR/datasets/OurHSI/gt"  # 替换为你的目录路径

    # 需要删除的通道索引（例如删除第 0 和第 2 通道）
    channels_to_remove = [465]

    # 处理目录下的所有 .npy 文件
    process_directory(directory, channels_to_remove)
