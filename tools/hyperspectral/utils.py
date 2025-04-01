import warnings
import numpy as np

warnings.simplefilter("always")  # 显示所有警告


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
            if channel.max() - channel.min() != 0:
                normalized_image[:, :, c] = (channel - channel.min()) / (channel.max() - channel.min())
            else:
                warnings.warn(f'normalize not used!, max:{channel.max()}, min: {channel.min()}')
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


def normalize(img, mode='normalize'):
    if mode == 'normalize':
        return normalize_image(img)
    elif mode == 'standard':
        return standardize_image(img)
    else:
        raise ValueError("mode 必须是 'normalize' 或 'standardize'")
