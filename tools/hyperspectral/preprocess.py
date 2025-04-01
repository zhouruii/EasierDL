import os
import numpy as np
import h5py

from utils import normalize


def spatial_cut(matrix: np.ndarray, block_size: int, zero_ratio_threshold: float = 0.06) -> list:
    """将矩阵的有效区域切割为指定大小的正方形块，并舍弃包含大量 0 的块。

    Args:
        matrix (np.ndarray): 形状为 (H, W, C) 的矩阵。
        block_size (int): 切割块的大小（正方形边长）。
        zero_ratio_threshold (float): 舍弃块的阈值，如果块中 0 的比例超过该值，则舍弃。

    Returns:
        list: 包含切割后的块及其位置信息的列表，每个元素为 (block, (x_start, y_start))。
    """

    # 获取有效区域的高度和宽度
    h, w, _ = matrix.shape

    # 计算切割块的起始位置
    blocks = []
    for y_start in range(0, h, block_size):
        for x_start in range(0, w, block_size):
            # 如果超出边界，向前移动裁剪窗口
            y_end = min(y_start + block_size, h)
            x_end = min(x_start + block_size, w)
            y_start = max(y_end - block_size, 0)
            x_start = max(x_end - block_size, 0)

            # 提取块
            block = matrix[y_start:y_end, x_start:x_end, :]

            # 检查块中 0 的比例
            zero_ratio = np.mean(block == 0)
            if zero_ratio > zero_ratio_threshold:
                continue

            # 保存块及其位置信息（相对于有效区域的坐标）
            blocks.append((block, (x_start, y_start)))

    return blocks


def remove_channels(mat_path, channels_to_remove):
    """
    从 .mat 文件中加载矩阵并剔除指定通道。

    Args:
        mat_path (str): .mat 文件路径。
        channels_to_remove (list): 需要剔除的通道索引（如 [0, 2]）。

    Returns:
        np.ndarray: 剔除通道后的矩阵。
    """
    # 加载 .mat 文件
    with h5py.File(mat_path, "r") as f:
        # 假设矩阵是 .mat 文件中的第一个变量
        key = list(f.keys())[0]
        matrix = np.array(f[key]).transpose()  # 转换为 (H, W, C) 格式

    # 确保矩阵是 HWC 格式
    if matrix.ndim != 3:
        raise ValueError(f"文件 {mat_path} 的矩阵维度不是 HWC")

    # 生成所有需要剔除的通道索引
    channels_to_drop = set()
    for start, end in channels_to_remove:
        channels_to_drop.update(range(start, end + 1))

    # 保留不需要剔除的通道
    channels_to_keep = [i for i in range(matrix.shape[2]) if i not in channels_to_drop]
    result_matrix = matrix[:, :, channels_to_keep]

    return result_matrix


def process_mat_files(input_dir, output_dir, channels_to_remove):
    """
    处理目录下的所有 .mat 文件，剔除指定通道并保存为 .npy 文件。

    Args:
        input_dir (str): 输入目录，包含 .mat 文件。
        output_dir (str): 输出目录，保存处理后的 .npy 文件。
        channels_to_remove (list): 需要剔除的通道索引。
    """
    os.makedirs(output_dir, exist_ok=True)

    # result[result < 0] = 0
    # result /= 10000
    # # result = result[:, :, rgb]
    # result = result[50:, 300:-200, :]
    # # result = normalize(result)
    # blocks = spatial_cut(result, block_size=512)
    # for idx, block in enumerate(blocks):
    #     block = normalize(block)
    #     np.save(f'data/{file}_{idx}.npy', block)

    # 遍历输入目录
    for filename in os.listdir(input_dir):
        if filename.endswith(".mat"):
            mat_path = os.path.join(input_dir, filename)
            print(f"processing: {mat_path}")

            try:
                # 剔除指定通道
                result_matrix = remove_channels(mat_path, channels_to_remove)
                result_matrix[result_matrix < 0] = 0

                blocks = spatial_cut(result_matrix, block_size=256)

                for block, (x_start, y_start) in blocks:
                    block = normalize(block)
                    npy_filename = filename.split(".")[0]
                    npy_filename = f'{npy_filename}_{y_start}_{x_start}.npy'
                    save_path = os.path.join(output_dir, npy_filename)
                    print(f'saving to {save_path}')
                    np.save(save_path, block)

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")


# 示例使用
if __name__ == "__main__":
    # 输入目录
    input_dir = "/home/disk1/ZR/datasets/OurHSI/2023年12月20日栗峪口村高光谱数据/900-2500nm/3"  # 替换为 .mat 文件目录

    # 输出目录
    output_dir = "/home/disk2/ZR/datasets/OurHSI/12-20/3"

    # 需要剔除的通道索引（如 [0, 2] 表示剔除第 0 和第 2 通道）
    channels_to_remove = [(1 + 272, 18 + 272), (79 + 272, 86 + 272), (156 + 272, 175 + 272), (242 + 272, 272 + 272)]

    # 处理 .mat 文件
    process_mat_files(input_dir, output_dir, channels_to_remove)
