import os.path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import minimum_filter


def dark_channel_prior(hyperspectral_data, window_size=3):
    """
    计算暗通道先验图（DCP）。
    每个像素取所有通道的最小值，再进行局部最小值滤波。
    """
    min_channels = np.min(hyperspectral_data, axis=2)  # H×W
    dcp_image = minimum_filter(min_channels, size=window_size, mode='reflect')
    dcp_image = (dcp_image - dcp_image.min()) / (dcp_image.max() - dcp_image.min()) * 255
    return dcp_image.astype(np.uint8)


def residual_channel_prior(hyperspectral_data):
    """
    计算残余通道先验图（RCP）。
    每个像素取所有通道的最大值减最小值。
    """
    H, W, C = hyperspectral_data.shape

    # 步骤1：计算每个通道的全局均值
    channel_means = np.mean(hyperspectral_data.reshape(-1, C), axis=0)  # 形状 [C]

    # 步骤2：按均值排序并分组（低均值组 vs 高均值组）
    sorted_indices = np.argsort(channel_means)  # 从小到大排序
    split_idx = C // 2
    low_group = sorted_indices[:split_idx]  # 低均值组通道索引
    high_group = sorted_indices[split_idx:]  # 高均值组通道索引

    # 步骤3：计算每个像素的组内均值差
    low_values = hyperspectral_data[:, :, low_group]  # 形状 [H, W, split_idx]
    high_values = hyperspectral_data[:, :, high_group]  # 形状 [H, W, C-split_idx]

    # 计算组内均值（沿通道维度）
    low_mean = np.mean(low_values, axis=2)  # 形状 [H, W]
    high_mean = np.mean(high_values, axis=2)  # 形状 [H, W]

    # 计算差值
    rcp_image = high_mean - low_mean  # 形状 [H, W]

    # # 滤波
    # rcp_image = torch.from_numpy(rcp_image)
    # blur_transform = GaussianBlur(kernel_size=9, sigma=(1.0, 2.0))  # sigma 可设为范围或固定值
    # rcp_image = blur_transform(rcp_image.unsqueeze(0)).squeeze(0).numpy()

    # 步骤5：归一化到 [0, 255]
    rcp_image = (rcp_image - rcp_image.min()) / (rcp_image.max() - rcp_image.min()) * 255
    return rcp_image.astype(np.uint8)


def mean_channel(hyperspectral_data):
    """
    计算高光谱数据的均值通道（每个像素点所有通道的平均值）。

    参数:
        hyperspectral_data: H×W×C 的高光谱数据 (np.ndarray)

    返回:
        mean_image: 单通道均值图 (H×W)
    """
    mean_image = np.mean(hyperspectral_data, axis=2)  # H×W
    # 归一化到 [0, 255]
    mean_image = (mean_image - mean_image.min()) / (mean_image.max() - mean_image.min()) * 255
    return mean_image.astype(np.uint8)


def get_haze_transmission(hyperspectral_data, split_band=100):
    """
    计算高光谱数据的均值通道（每个像素点所有通道的平均值）。

    参数:
        hyperspectral_data: H×W×C 的高光谱数据 (np.ndarray)

    返回:
        mean_image: 单通道均值图 (H×W)
    """
    heavy, light = hyperspectral_data[:, :, :split_band], hyperspectral_data[:, :, split_band:]
    heavy = np.mean(heavy, axis=2)
    light = np.mean(light, axis=2)
    t = heavy - light  # H×W
    # # 滤波
    # t = torch.from_numpy(t)
    # blur_transform = GaussianBlur(kernel_size=9, sigma=(1.0, 2.0))  # sigma 可设为范围或固定值
    # t = blur_transform(t.unsqueeze(0)).squeeze(0).numpy()
    # 归一化到 [0, 255]
    t = (t - t.min()) / (t.max() - t.min()) * 255
    return t.astype(np.uint8)


def visualize_prior_maps(*args):
    """
    可视化原始数据、DCP和RCP图，并即时显示。
    """
    nums = len(args)
    # RGB
    rgb_image = args[0][:, :, [36, 19, 8]]
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())  # 归一化到 [0,1]

    plt.figure(figsize=(15, 5))

    # 原始RGB图像
    plt.subplot(1, nums, 1)
    plt.imshow(rgb_image)
    plt.title("LQ RGB", fontsize=14)
    plt.axis('off')

    rgb_image = args[1][:, :, [36, 19, 8]]
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())  # 归一化到 [0,1]
    plt.subplot(1, nums, 2)
    plt.imshow(rgb_image)
    plt.title("GT RGB", fontsize=14)
    plt.axis('off')

    for i in range(2, nums):
        plt.subplot(1, nums, i + 1)
        plt.imshow(args[i], cmap='gray')
        plt.colorbar(fraction=0.03, pad=0.04)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('prior.png')
    plt.show()


if __name__ == '__main__':
    filename = 'f130803t01p00r15rdn_e_16.npy'
    data_path = os.path.join('/home/disk2/ZR/datasets/AVIRIS/512/rain/storm', filename)
    gt_path = os.path.join('/home/disk2/ZR/datasets/AVIRIS/512/gt', filename)
    # data_path = '/home/disk2/ZR/datasets/AVIRIS/512/train/rain/storm/f130804t01p00r04rdn_e_11.npy'
    # gt_path = '/home/disk2/ZR/datasets/AVIRIS/512/train/gt/f130804t01p00r04rdn_e_11.npy'
    data = np.load(data_path)
    gt = np.load(gt_path)
    # 计算DCP和RCP
    # dcp = dark_channel_prior(data, window_size=3)
    sorted_rcp = residual_channel_prior(data)
    # mean = mean_channel(data)
    transmission = get_haze_transmission(data, split_band=190)

    # 即时可视化
    visualize_prior_maps(data, gt, sorted_rcp, transmission)
