import cv2
import matplotlib.pyplot as plt
import numpy as np

from tools.synthesis.gen_perlin import generate_perlin_noise
from util import read_img, to_visualize


def guided_filter_single_channel(guide_image, input_image, radius, epsilon):
    """ `Guided image filter` implementation using OpenCV.
    
    Args:
        guide_image (): Guide image (e.g., clean image).
        input_image (): Input image to be filtered (e.g., rain or fog intensity map).
        radius (): The radius of the local window.
        epsilon (): Regularization parameter, controls smoothness.

    Returns:

    """

    # Ensure the images are in float32 format for better precision
    guide_image = guide_image.astype(np.float32)
    input_image = input_image.astype(np.float32)

    # Compute the mean of guide and input
    mean_guide = cv2.boxFilter(guide_image, -1, (radius, radius))
    mean_input = cv2.boxFilter(input_image, -1, (radius, radius))

    # Compute the correlation between guide and input
    mean_guide_input = cv2.boxFilter(guide_image * input_image, -1, (radius, radius))

    # Compute the variance of the guide
    mean_guide_squared = cv2.boxFilter(guide_image * guide_image, -1, (radius, radius))
    var_guide = mean_guide_squared - mean_guide * mean_guide

    # Compute the coefficients a and b
    a = (mean_guide_input - mean_guide * mean_input) / (var_guide + epsilon)
    b = mean_input - a * mean_guide

    # Compute the output
    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))

    output_image = mean_a * guide_image + mean_b
    return output_image


def guided_filter(guide_image=None, input_image=None, radius=None, epsilon=None):

    out = np.zeros(guide_image.shape)
    if len(guide_image.shape) == 3:
        g_h, g_w, g_c = guide_image.shape
    else:
        g_h, g_w = guide_image.shape
        g_c = 0

    if len(input_image.shape) != len(guide_image.shape):
        input_image = np.expand_dims(input_image, axis=-1)  # [H, W, 1]
        input_image = np.tile(input_image, (1, 1, g_c))

    if g_c != 0:
        for c in range(g_c):
            out[:, :, c] = guided_filter_single_channel(guide_image[:, :, c], input_image[:, :, c], radius, epsilon)
    else:
        out = guided_filter_single_channel(guide_image, input_image, radius, epsilon)

    return out


if __name__ == '__main__':
    # 读取图像
    # guide_img = read_img('demo/5.jpg', scale=False)  # 引导图像
    guide_img = cv2.imread('demo/5.jpg', cv2.IMREAD_GRAYSCALE)  # 引导图像
    height, width = guide_img.shape[0], guide_img.shape[1]
    input_img = generate_perlin_noise(height=height, width=width, scales=[300])  # 需要滤波的目标图像

    # 调用引导滤波器函数
    r = 8  # 滤波窗口的半径
    epsilon = 0.1  # 正则化项
    output = guided_filter(
        guide_image=guide_img,
        input_image=input_img,
        radius=r,
        epsilon=epsilon)

    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    axes[0].imshow(to_visualize(guide_img), cmap='gray')
    axes[1].imshow(to_visualize(input_img), cmap='gray')
    axes[2].imshow(to_visualize(output), cmap='gray')

    plt.tight_layout()
    plt.show()
