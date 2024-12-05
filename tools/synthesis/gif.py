import numpy as np
import cv2
from scipy.ndimage import uniform_filter

from util import resize_image


def guided_filter_cv2(guide_image, input_image, radius, epsilon):
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


def guided_filter(impl='cv2', guide_image=None, input_image=None, radius=None, epsilon=None):
    g_h, g_w = guide_image.shape
    input_image = resize_image(image=input_image, new_width=g_w, new_height=g_h)
    if impl == 'cv2':
        return guided_filter_cv2(guide_image, input_image, radius, epsilon)
    else:
        raise NotImplementedError(f'{impl} is not supported yet!')


if __name__ == '__main__':
    # 读取图像
    guide_image = cv2.imread('./demo/BSD300/22013.jpg', cv2.IMREAD_GRAYSCALE)  # 引导图像
    input_image = cv2.imread('./demo/Streaks_Garg06/1-5.png', cv2.IMREAD_GRAYSCALE)  # 需要滤波的目标图像

    # 调用引导滤波器函数
    r = 6  # 滤波窗口的半径
    epsilon = 0.1  # 正则化项
    output = guided_filter(
        impl='cv2',
        guide_image=guide_image,
        input_image=input_image,
        radius=r,
        epsilon=epsilon)

    cv2.imwrite('./demo/filter.jpg', output)
