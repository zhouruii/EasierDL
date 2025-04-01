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
            rgb_data = data[:, :, [136, 67, 18]]
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


def get_file_pairs(clean_dir, noise_dirs):
    """
    获取干净数据和噪声数据的文件对。

    Args:
        clean_dir (str): 干净数据目录。
        noise_dirs (list): 噪声数据目录列表。

    Returns:
        list: 文件对列表，每个元素为 (clean_path, noise_paths)。
    """
    file_pairs = []

    # 遍历干净数据目录
    for clean_filename in os.listdir(clean_dir):
        if clean_filename.endswith(".npy"):
            clean_path = os.path.join(clean_dir, clean_filename)
            key = clean_filename.replace(".npy", "")

            # 查找对应的噪声文件
            noise_paths = []
            for noise_dir in noise_dirs:
                noise_filename = f"{key}_*.npy"
                noise_files = [f for f in os.listdir(noise_dir) if f.startswith(f'{key}_') and f.endswith(".npy")]
                if noise_files:
                    noise_paths.append(os.path.join(noise_dir, noise_files[0]))

            if len(noise_paths) == 4:  # 确保有四个噪声文件
                file_pairs.append((clean_path, noise_paths))

    return file_pairs


def select_rgb_channels(image, channels):
    """
    选取指定通道生成 RGB 图像。

    Args:
        image (np.ndarray): 输入图像，形状为 (H, W, C)。
        channels (list): 选择的通道索引（如 [0, 1, 2]）。

    Returns:
        np.ndarray: RGB 图像，形状为 (H, W, 3)。
    """
    return image[:, :, channels]


def visualize_and_save(clean_path, noise_paths, channels, output_dir):
    """
    可视化并保存五张图片（一张干净数据和四张噪声数据）。

    Args:
        clean_path (str): 干净数据文件路径。
        noise_paths (list): 噪声数据文件路径列表。
        channels (list): 选择的通道索引。
        output_dir (str): 输出图片保存目录。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 读取干净数据
    clean_data = np.load(clean_path)
    clean_rgb = select_rgb_channels(clean_data, channels)

    # 读取噪声数据
    noise_rgb_list = []
    for noise_path in noise_paths:
        noise_data = np.load(noise_path)
        noise_rgb = select_rgb_channels(noise_data, channels)
        noise_rgb_list.append(noise_rgb)

    # 可视化
    plt.figure(figsize=(15, 5))
    # titles = ["Clean Data"] + [f"Noise Level {i + 1}" for i in range(4)]
    titles = ["Clean", "Small", "Medium", "Heavy", "Storm"]
    for i, (title, img) in enumerate(zip(titles, [clean_rgb] + noise_rgb_list)):
        plt.subplot(1, 5, i + 1)
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")

    # 保存结果
    filename = os.path.basename(clean_path).replace(".npy", "_comparison.png")
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"preview result is saved to --> {output_path}")


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
    if path.endswith('.npy'):
        return np.load(path)
    with rasterio.open(path) as src:
        # 读取数据
        data = src.read()  # 数据形状为 (波段数, 行数, 列数)
        print("数据形状:", data.shape)
        bands = src.descriptions
        bands = [float(item.split()[0]) for item in bands]
        bands = np.array(bands)

    return data


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
    # rgb = [36, 19, 8]
    # file = 'f130804t01p00r06rdn_e'
    # file_path = f"/home/disk1/ZR/datasets/AVIRIS/raw/{file}/{file}_sc01_ort_img.hdr"
    # result = read_hdr(file_path, load_hsi=True)['hsi']
    # result[result < 0] = 0
    # result /= 10000
    # # result = result[:, :, rgb]
    # result = result[50:, 300:-200, :]
    # # result = normalize(result)
    # blocks = spatial_cut(result, block_size=512)
    # for idx, block in enumerate(blocks):
    #     block = normalize(block)
    #     np.save(f'data/{file}_{idx}.npy', block)

    # rgb = [136, 67, 18]
    # file_path = "/home/disk1/ZR/datasets/OurHSI/4/raw_0_rd_rf_or"
    # rgb = [136+272, 67+272, 18+272]
    # file_path = "/home/disk2/ZR/datasets/merged_0_0.npy"
    # result = read_data(file_path)  # (C, H, W)
    # result[result < 0] = 0
    # result = np.transpose(result, (1, 2, 0))
    # result = result[:, :, rgb]
    #
    # result = normalize(np.clip(result, 0, 1))
    #
    # io.imsave(f"demo2.jpg", (result * 255).astype(np.uint8))

    # process_npy_files('/home/disk2/ZR/datasets/OurHSI/12-20/2', '/home/disk2/ZR/datasets/OurHSI/preview/12-20/2')

    # result = read_hdr(file_path, load_hsi=False)['meta']
    # bands = result.get('wavelength')
    # bands = [float(band) for band in bands]
    # bands = np.array(bands)
    # result['bands'] = bands
    # with open('meta.pkl', "wb") as f:
    #     pickle.dump(result, f)

    # # 输入目录
    # clean_dir = "/home/disk2/ZR/datasets/AVIRIS/512/gt"  # 替换为干净数据目录
    # noise_dirs = [
    #     "/home/disk2/ZR/datasets/AVIRIS/512/rain/small",  # 替换为噪声级别1的目录
    #     "/home/disk2/ZR/datasets/AVIRIS/512/rain/medium",  # 替换为噪声级别2的目录
    #     "/home/disk2/ZR/datasets/AVIRIS/512/rain/heavy",  # 替换为噪声级别3的目录
    #     "/home/disk2/ZR/datasets/AVIRIS/512/rain/storm"  # 替换为噪声级别4的目录
    # ]
    #
    # # 输出目录
    # output_dir = "/home/disk2/ZR/datasets/AVIRIS/512/preview"
    #
    # # 选择通道（例如 [0, 1, 2] 表示前三个通道）
    # channels = [36, 19, 8]
    #
    # # 获取文件对
    # file_pairs = get_file_pairs(clean_dir, noise_dirs)
    #
    # # 依次处理每个文件对
    # for clean_path, noise_paths in file_pairs:
    #     visualize_and_save(clean_path, noise_paths, channels, output_dir)

    # 输入目录
    clean_dir = "/home/disk2/ZR/datasets/OurHSI/gt"  # 替换为干净数据目录
    noise_dirs = [
        "/home/disk2/ZR/datasets/OurHSI/rain/small",  # 替换为噪声级别1的目录
        "/home/disk2/ZR/datasets/OurHSI/rain/medium",  # 替换为噪声级别2的目录
        "/home/disk2/ZR/datasets/OurHSI/rain/heavy",  # 替换为噪声级别3的目录
        "/home/disk2/ZR/datasets/OurHSI/rain/storm"  # 替换为噪声级别4的目录
    ]

    # 输出目录
    output_dir = "/home/disk2/ZR/datasets/OurHSI/preview"

    # 选择通道（例如 [0, 1, 2] 表示前三个通道）
    channels = [136, 67, 18]

    # 获取文件对
    file_pairs = get_file_pairs(clean_dir, noise_dirs)

    # 依次处理每个文件对
    for clean_path, noise_paths in file_pairs:
        visualize_and_save(clean_path, noise_paths, channels, output_dir)
