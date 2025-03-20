import os
import pickle

import numpy as np
import rasterio
import spectral as sp
from skimage import io
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import normalize


def visualize_and_save_npy(file_path, output_dir):
    """
    加载 .npy 文件，可视化并保存结果。

    Args:
        file_path (str): .npy 文件的路径。
        output_dir (str): 保存可视化结果的目录。
    """
    # 加载 .npy 文件
    data = np.load(file_path)

    # 检查数据维度
    if data.ndim not in [2, 3]:
        print(f"文件 {file_path} 的数据维度不支持可视化: {data.shape}")
        return

    # 可视化数据
    plt.figure(figsize=(10, 10))
    if data.ndim == 2:
        # 二维数据：灰度图
        plt.imshow(data, cmap='gray')
        plt.title(f"2D Data: {os.path.basename(file_path)}")
    elif data.ndim == 3:
        # 三维数据：伪彩色图（取前三个波段作为 RGB）
        if data.shape[2] >= 3:
            rgb_data = data[:, :, [36, 19, 8]]
            plt.imshow(rgb_data)
            plt.title(f"3D Data (RGB): {os.path.basename(file_path)}")
        else:
            print(f"文件 {file_path} 的波段数不足，无法生成伪彩色图: {data.shape}")
            return

    # 保存可视化结果
    output_path = os.path.join(output_dir, os.path.basename(file_path).replace('.npy', '.png'))
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"可视化结果已保存到: {output_path}")


def process_npy_files(directory, output_dir):
    """
    处理目录中的所有 .npy 文件。

    Args:
        directory (str): 包含 .npy 文件的目录。
        output_dir (str): 保存可视化结果的目录。
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            file_path = os.path.join(directory, filename)
            print(f"正在处理文件: {file_path}")
            visualize_and_save_npy(file_path, output_dir)


def find_valid_region(matrix: np.ndarray) -> tuple:
    """找到矩阵的有效区域（去除边缘的 0）。

    Args:
        matrix (np.ndarray): 形状为 (C, H, W) 的矩阵。

    Returns:
        tuple: 有效区域的边界坐标 (x_min, x_max, y_min, y_max)。
    """
    # 检查所有通道是否非零
    non_zero_mask = np.all(matrix != 0, axis=2)

    # 找到所有通道都非零的区域
    non_zero_indices = np.where(non_zero_mask)

    # 如果所有像素都为零，返回整个区域的边界
    if len(non_zero_indices) == 0:
        return 0, matrix.shape, 0, matrix.shape

    # 找到有效区域的边界
    row_min, row_max = np.min(non_zero_indices[0]), np.max(non_zero_indices[0]) + 1
    col_min, col_max = np.min(non_zero_indices[1]), np.max(non_zero_indices[1]) + 1

    return row_min, row_max, col_min, col_max


def spatial_cut(matrix: np.ndarray, block_size: int) -> list:
    """将矩阵的有效区域切割为指定大小的正方形块。

    Args:
        matrix (np.ndarray): 形状为 (C, H, W) 的矩阵。
        block_size (int): 切割块的大小（正方形边长）。

    Returns:
        list: 包含切割后的块及其位置信息的列表，每个元素为 (block, (x_start, y_start))。
    """
    # 找到有效区域
    row_min, row_max, col_min, col_max = find_valid_region(matrix)
    valid_region = matrix[row_min:row_max, col_min:col_max, :]

    # 获取有效区域的高度和宽度
    h, w, _ = valid_region.shape

    # 计算切割块的起始位置
    blocks = []
    for y_start in range(0, h, block_size):
        for x_start in range(0, w, block_size):
            y_end = y_start + block_size
            x_end = x_start + block_size

            # 如果超出边界，舍弃
            if y_end > h or x_end > w:
                # block = np.zeros((matrix.shape, block_size, block_size), dtype=matrix.dtype)
                # block[:, :min(block_size, h - y_start), :min(block_size, w - x_start)] = \
                #     valid_region[:, y_start:y_end, x_start:x_end]
                continue
            else:
                block = valid_region[y_start:y_end, x_start:x_end, :]

            # 保存块及其位置信息
            blocks.append(block)

    return blocks


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

    # bands_centers = data.bands.centers
    # num_bands = data.nbands
    # data_shape = data.shape
    # gsd = data.metadata['map info'][5]
    # print(num_bands, data_shape, gsd)
    #
    # result = {'meta': data.metadata,
    #           'hsi_shape': data_shape,
    #           'bands_centers': bands_centers,
    #           'num_bands': num_bands,
    #           'gsd': gsd}

    result = {'meta': data.metadata,
              'hsi_shape': data.shape,
              }

    if load_hsi:
        if rows and cols:
            hsi = data.read_subregion(row_bounds=rows, col_bounds=cols)
        else:
            hsi = np.array(data.load())
        result['hsi'] = hsi

    return result


if __name__ == '__main__':
    # rgb = [63, 34, 12]
    file = 'f130411t01p00r11rdn_e'
    file_path = f"/home/disk1/ZR/datasets/AVIRIS/raw/{file}/{file}_sc01_ort_img.hdr"

    rows = [i for i in range(0, 3000, 512)]

    # for row in rows:
    #     result = read_hdr(file_path, load_hsi=True, rows=(row, row + 512), cols=(0, 512))['hsi']
    #     # result = read_hdr(file_path, load_hsi=True)['hsi']
    #     result[result < 0] = 0
    #     row_min, row_max, col_min, col_max = find_valid_region(result)
    #     result = result[:, :, rgb]
    #
    #     result = normalize(np.clip(result, 0, 1))
    #
    #     io.imsave(f"demo/{row}.jpg", (result * 255).astype(np.uint8))

    # rgb = [36, 19, 8]
    # result = read_hdr(file_path, load_hsi=True)['hsi']
    # result[result < 0] = 0
    # result /= 10000
    # # result = result[:, :, rgb]
    # result = result[25:, 100:-100, :]
    # # result = normalize(result)
    #
    # blocks = spatial_cut(result, block_size=512)
    # for idx, block in enumerate(blocks):
    #     block = normalize(block)
    #     np.save(f'data/{file}_{idx}.npy', block)

    process_npy_files('data', 'preview')

    # result = read_hdr(file_path, load_hsi=False)['meta']
    # bands = result.get('wavelength')
    # bands = [float(band) for band in bands]
    # bands = np.array(bands)
    # result['bands'] = bands
    # with open('meta.pkl', "wb") as f:
    #     pickle.dump(result, f)
