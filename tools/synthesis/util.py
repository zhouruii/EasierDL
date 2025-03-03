import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr_val
from skimage.metrics import structural_similarity as ssim_val


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


def to_visualize(img):
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


def normalize(img, _min, _max):
    return cv2.normalize(img, None, _min, _max, cv2.NORM_MINMAX)


def check_dtype(data):
    if data.dtype == np.uint8:
        data = data / 255
        return data.astype(np.float32)


def visualize_tool(fig_size=None,
                   rows_cols=None,
                   data_dict=None):
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
    return streak[top_left_y:top_left_y+crop_height, top_left_x:top_left_x+crop_width]


if __name__ == '__main__':
    img1 = read_img(r"E:\datasets\test\clean\5.jpg")
    img2 = read_img(r"E:\datasets\test\haze\5.jpg")
    psnr_val, ssim_val = calculate_psnr_ssim(img1, img2)
    print(f'PSNR:{psnr_val}, SSIM:{ssim_val}')
