import os
import random

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr_val
from skimage.metrics import structural_similarity as ssim_val
from sklearn.decomposition import PCA


def resize_image(image=None, scale_factor=None, new_height=None, new_width=None):
    """
    :param image: 输入图像的路径
    :param scale_factor: 缩放因子
    :return: 缩放后的图像
    """

    if scale_factor is not None:
        assert new_width is None and new_height is None
    if scale_factor is None:
        assert new_width is not None and new_height is not None

    # 读取图像
    if isinstance(image, str):
        image = cv2.imread(image)
    else:
        image = image

    if image is None:
        print("Error: Image not found.")
        return None

    # 获取原图像的尺寸
    original_height, original_width = image.shape[:2]

    # 计算新的尺寸
    if scale_factor is not None:
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
    else:
        new_width = new_width
        new_height = new_height

    # 使用OpenCV的resize函数来缩放图像
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return resized_image


def read_img(path, to_rgb=True, scale=True, dtype='float32'):
    img = cv2.imread(path)

    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if dtype == 'float32':
        img = img.astype(np.float32)

    if scale:
        img = img / 255.0

    return img


def normalize_image(image: np.ndarray) -> np.ndarray:
    """对图片进行归一化（缩放到 [0, 1] 范围）。

    Args:
        image (np.ndarray): 输入图片，形状为 (H, W) 或 (C, H, W)。

    Returns:
        np.ndarray: 归一化后的图片。
    """
    if image.ndim == 2:  # 单通道图片
        return (image - image.min()) / (image.max() - image.min())
    elif image.ndim == 3:  # 多通道图片
        normalized_image = np.zeros_like(image, dtype=np.float32)
        for c in range(image.shape[-1]):  # 对每个通道单独归一化
            channel = image[:, :, c]
            normalized_image[:, :, c] = (channel - channel.min()) / (channel.max() - channel.min())
        return normalized_image
    else:
        raise ValueError("输入图片的形状必须是 (H, W) 或 (H, W, C)")


def standardize_image(image: np.ndarray) -> np.ndarray:
    """对图片进行标准化（均值为 0，标准差为 1）。

    Args:
        image (np.ndarray): 输入图片，形状为 (H, W) 或 (C, H, W)。

    Returns:
        np.ndarray: 标准化后的图片。
    """
    if image.ndim == 2:  # 单通道图片
        return (image - image.mean()) / image.std()
    elif image.ndim == 3:  # 多通道图片
        standardized_image = np.zeros_like(image, dtype=np.float32)
        for c in range(image.shape[-1]):  # 对每个通道单独标准化
            channel = image[:, :, c]
            standardized_image[:, :, c] = (channel - channel.mean()) / channel.std()
        return standardized_image
    else:
        raise ValueError("输入图片的形状必须是 (H, W) 或 (H, W, C)")


def to_visualize(img, bands=[136, 67, 18]):
    if len(img.shape) == 3 and img.shape[2] > 3:
        return to_visualize_hsi(img, bands)
    if img.dtype == np.uint8:
        return img

    # img = normalize(img, 0, 255)
    img = np.clip(img, 0, 1.0) * 255

    if img.dtype == np.float64 or img.dtype == np.float32:
        img = img.astype(np.uint8)

    return img


def calculate_psnr_ssim(img1: np.ndarray, img2: np.ndarray) -> tuple:
    """
    计算 PSNR 和 SSIM。

    参数:
        img1: ndarray, 输入图像 1，形状 [H, W, C]
        img2: ndarray, 输入图像 2，形状 [H, W, C]

    返回:
        PSNR: float, 峰值信噪比
        SSIM: float, 结构相似性
    """
    assert img1.shape == img2.shape, "两个图像的形状必须相同"

    # 逐通道计算 SSIM，然后求平均值
    channels = img1.shape[2]
    ssim_values = [
        ssim_val(img1[..., i], img2[..., i], data_range=img1[..., i].max() - img1[..., i].min())
        for i in range(channels)
    ]
    avg_ssim = np.mean(ssim_values)

    # PSNR：直接支持多通道
    psnr_value = psnr_val(img1, img2, data_range=img1.max() - img1.min())

    return psnr_value, avg_ssim


def normalize(img, mode='normalize'):
    if mode == 'normalize':
        return normalize_image(img)
    elif mode == 'standard':
        return standardize_image(img)
    else:
        raise ValueError("mode 必须是 'normalize' 或 'standardize'")


def check_dtype(data):
    if data.dtype == np.uint8:
        data = data / 255
        return data.astype(np.float32)


def visualize_tool(fig_size=None,
                   rows_cols=None,
                   data_dict=None,
                   save_path=False):
    fig, axes = plt.subplots(*rows_cols, figsize=fig_size)

    for row in range(rows_cols[0]):
        for col in range(rows_cols[1]):
            label, data = data_dict.popitem(last=False)
            if len(data.shape) == 2:
                axes[row, col].imshow(to_visualize(data), cmap='gray')
            else:
                axes[row, col].imshow(to_visualize(data))
            axes[row, col].set_title(label)
            axes[row, col].axis('off')

    if save_path is not None:
        # os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path)

    plt.tight_layout()
    plt.show()


def scale_streak(streak):
    height, width = streak.shape[:2]
    new_streak = resize_image(streak, scale_factor=1.5)
    new_height, new_width = new_streak.shape[:2]
    top_left = ((new_height - height) // 2, (new_width - width) // 2)
    if len(streak.shape) == 3:
        scaled_streak = new_streak[top_left[0]:top_left[0] + height, top_left[1]:top_left[1] + width, :]
    else:
        scaled_streak = new_streak[top_left[0]:top_left[0] + height, top_left[1]:top_left[1] + width]
    return scaled_streak


def crop_streak(streak, crop_size):
    height, width = streak.shape[:2]
    crop_height, crop_width = crop_size
    top_left_y = (height - crop_height) // 2
    top_left_x = (width - crop_width) // 2
    return streak[top_left_y:top_left_y + crop_height, top_left_x:top_left_x + crop_width]


def get_random_image(folder_path: str, mode: str = 'gray') -> np.ndarray:
    """从文件夹中随机读取一张图片。

    Args:
        folder_path (str): 图片文件夹的路径。
        mode(str): 读取模式，灰度或彩色

    Returns:
        Image.Image: 随机读取的图片对象。

    Raises:
        FileNotFoundError: 如果文件夹不存在或文件夹中没有图片。
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"文件夹 {folder_path} 不存在")

    # 获取文件夹中所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']  # 支持的图片格式
    image_files = [file for file in os.listdir(folder_path) if os.path.splitext(file)[-1].lower() in image_extensions]

    # 检查文件夹中是否有图片
    if not image_files:
        raise FileNotFoundError(f"文件夹 {folder_path} 中没有图片")

    # 随机选择一张图片
    random_image_file = random.choice(image_files)
    image_path = os.path.join(folder_path, random_image_file)

    # 打开并返回图片
    if mode == 'rgb':
        image = cv2.imread(image_path)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image


def to_visualize_hsi(hsi, bands=[136, 67, 18]):
    if bands is not None:
        hsi = hsi[:, :, bands]
    else:
        # 将高光谱数据重塑为 (像素数, 波段数)
        h, w, bands = hsi.shape[0], hsi.shape[1], hsi.shape[2]
        pixels = hsi.reshape(bands, -1).T  # 形状为 (像素数, 波段数)

        # 使用PCA降维到3个主成分
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(pixels)  # 形状为 (像素数, 3)

        # 将PCA结果重塑为图像
        hsi = pca_result.reshape(h, w, 3)
    # hsi = normalize(hsi) * 255
    # cv2.normalize(hsi, hsi, 0, 1)
    # hsi *= 255
    hsi = np.clip(hsi, 0, 1.0) * 255
    return hsi.astype(np.uint8)


if __name__ == '__main__':
    img1 = read_img(r"E:\datasets\test\clean\5.jpg")
    img2 = read_img(r"E:\datasets\test\haze\5.jpg")
    psnr_val, ssim_val = calculate_psnr_ssim(img1, img2)
    print(f'PSNR:{psnr_val}, SSIM:{ssim_val}')
