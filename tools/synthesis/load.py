import pickle
from typing import Optional

import cv2
import rasterio
import spectral as sp
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
from skimage import io

from util import normalize


def save_pickle(save_path, save_data):
    with open(save_path, "wb") as f:
        pickle.dump(save_data, f)
    print(f"所有信息已保存到: {save_path}")


def load_pickle(pickle_path):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return data


def load_mat(mat_path):
    return loadmat(mat_path)['hsi_0']


def find_valid_region(matrix: np.ndarray) -> tuple:
    """找到矩阵的有效区域（去除边缘的 0）。

    Args:
        matrix (np.ndarray): 形状为 (C, H, W) 的矩阵。

    Returns:
        tuple: 有效区域的边界坐标 (x_min, x_max, y_min, y_max)。
    """
    # 检查所有通道是否非零
    non_zero_mask = np.all(matrix != 0, axis=0)

    # 找到所有通道都非零的区域
    non_zero_indices = np.where(non_zero_mask)

    # 如果所有像素都为零，返回整个区域的边界
    if len(non_zero_indices) == 0:
        return 0, matrix.shape, 0, matrix.shape

    # 找到有效区域的边界
    y_min, y_max = np.min(non_zero_indices), np.max(non_zero_indices) + 1
    x_min, x_max = np.min(non_zero_indices), np.max(non_zero_indices) + 1

    return x_min, x_max, y_min, y_max


def spatial_cut(matrix: np.ndarray, block_size: int) -> list:
    """将矩阵的有效区域切割为指定大小的正方形块。

    Args:
        matrix (np.ndarray): 形状为 (C, H, W) 的矩阵。
        block_size (int): 切割块的大小（正方形边长）。

    Returns:
        list: 包含切割后的块及其位置信息的列表，每个元素为 (block, (x_start, y_start))。
    """
    # 找到有效区域
    x_min, x_max, y_min, y_max = find_valid_region(matrix)
    valid_region = matrix[:, y_min:y_max, x_min:x_max]

    # 获取有效区域的高度和宽度
    _, h, w = valid_region.shape

    # 计算切割块的起始位置
    blocks = []
    for y_start in range(0, h, block_size):
        for x_start in range(0, w, block_size):
            y_end = y_start + block_size
            x_end = x_start + block_size

            # 如果超出边界，则填充 0
            if y_end > h or x_end > w:
                block = np.zeros((matrix.shape, block_size, block_size), dtype=matrix.dtype)
                block[:, :min(block_size, h - y_start), :min(block_size, w - x_start)] = \
                    valid_region[:, y_start:y_end, x_start:x_end]
            else:
                block = valid_region[:, y_start:y_end, x_start:x_end]

            # 保存块及其位置信息
            blocks.append((block, (x_start + x_min, y_start + y_min)))

    return blocks


def find_zero_channels(matrix: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    """找出矩阵中基本全为 0 的通道索引。

    Args:
        matrix (np.ndarray): 形状为 (C, H, W) 的矩阵。
        threshold (float): 判断通道是否全为 0 的阈值，默认为 1e-6。

    Returns:
        np.ndarray: 基本全为 0 的通道索引。
    """
    # 计算每个通道的像素值之和
    channel_sums = np.sum(np.abs(matrix), axis=(1, 2))

    # 找出像素值之和小于阈值的通道索引
    zero_channels = np.where(channel_sums < threshold)

    return zero_channels


def process_mat_to_rgb(file_path: str, band_indices: tuple = (0, 1, 2), data_key: Optional[str] = None) -> None:
    """从 .mat 文件中读取数据，提取指定波段生成 RGB 图像并可视化。

    Args:
        file_path (str): .mat 文件路径。
        band_indices (tuple, optional): 波段索引，默认为 (0, 1, 2)。
        data_key (Optional[str], optional): .mat 文件中 HWC 数组的键名。如果为 None，则自动猜测第一个数组。

    Returns:
        None: 该函数直接显示图像，不返回任何值。
    """
    # 1. 加载 .mat 文件
    data = loadmat(file_path)

    # 2. 获取 HWC 数组
    if data_key is None:
        # 如果未指定键名，猜测第一个数组
        data_key = [key for key in data if not key.startswith('__')]
    image_data = data[data_key]

    # 3. 提取指定波段并归一化到 [0, 1]
    rgb_image = np.stack(
        [image_data[:, :, idx] / image_data[:, :, idx].max() for idx in band_indices],
        axis=-1
    )

    plt.imsave('demo/2_12_4_1.jpg', rgb_image)


def read_data(path):
    with rasterio.open(path) as src:
        # 读取数据
        data = src.read()  # 数据形状为 (波段数, 行数, 列数)
        print("数据形状:", data.shape)
        bands = src.descriptions
        bands = [float(item.split()[0]) for item in bands]
        bands = np.array(bands)

    return bands


def read_hdr(path, load_hsi=False, rows=None, cols=None):
    data = sp.open_image(path)

    bands_centers = data.bands.centers
    num_bands = data.nbands
    data_shape = data.shape
    gsd = data.metadata['map info'][5]
    print(num_bands, data_shape, gsd)

    result = {'meta': data.metadata,
              'hsi_shape': data_shape,
              'bands_centers': bands_centers,
              'num_bands': num_bands,
              'gsd': gsd}

    if load_hsi:
        if rows and cols:
            hsi = data.read_subregion(row_bounds=rows, col_bounds=cols)
        else:
            hsi = data.load()
        result['hsi'] = hsi

    return result


if __name__ == '__main__':
    # remove_ranges = [(1, 18), (79, 86), (156, 175), (242, 272)]
    # bands1 = read_data(
    #     "/home/disk1/ZR/datasets/OurHSI/20231220-21西电栗峪口村高光谱数据/12-21/400-1000nm/raw_17120_rd_rf_or")
    # bands2 = read_data(
    #     "/home/disk1/ZR/datasets/OurHSI/20231220-21西电栗峪口村高光谱数据/12-21/900-2500nm/raw_3150_rd_rf_or")
    # # 创建一个掩码，标记需要保留的元素
    # mask = np.ones(len(bands2), dtype=bool)
    # for start, end in remove_ranges:
    #     mask[start-1:end] = False  # 去除指定区域
    #
    # # 使用掩码提取保留的部分
    # result = bands2[mask]
    # save_data = {
    #     "bands": np.concatenate([bands1, result]),
    #     "total_bands": np.concatenate([bands1, bands2]),
    #     "bands1(400~1000/nm)": bands1,
    #     "bands2(900~2500/nm)": bands2,
    #     "remove_ranges_of_bands2": remove_ranges
    # }
    # save_pickle("bands.pkl", save_data)

    # rgb = [53, 32, 13]
    rgb = [63, 34, 12]
    file_path = "/home/disk1/ZR/datasets/AVIRIS/ang20191021t151200_rfl_v2x1/ang20191021t151200_corr_v2x1_img.hdr"

    rows = [i for i in range(1000, 3000, 512)]

    for row in rows:
        result = read_hdr(file_path, load_hsi=True, rows=(row, row + 512), cols=(100, 612))['hsi']
        result = result[:, :, rgb]
        result[result < 0] = 0
        result = normalize(np.clip(result, 0, 1))

        io.imsave(f"demo/{row}.jpg", (result * 255).astype(np.uint8))

    # mat_file_path = 'demo/2_12_1_1.mat'
    # process_mat_to_rgb(mat_file_path, band_indices=tuple(rgb), data_key='hsi_0')
