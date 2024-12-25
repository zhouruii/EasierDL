import cv2
import numpy as np
from matplotlib import pyplot as plt

from tools.synthesis.util import read_img, to_visualize, resize_image


def generate_rain_accumulation_mask(shape, centers, sigma, alpha):
    """
    生成雨流积聚的散射系数分布。

    参数:
    shape - 图像大小 (tuple)
    centers - 雨流中心点的坐标列表 [(x1, y1), (x2, y2), ...]
    sigma - 高斯分布的标准差 (scalar)
    alpha - 散射强度比例因子 (scalar)

    返回:
    gamma_rain_accum - 雨积聚的散射系数分布 (2D array)
    """
    gamma_rain_accum = np.zeros(shape)
    C = shape[-1]
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    for c in range(C):
        for cx, cy in centers:
            gamma_rain_accum[:, :, c] += alpha * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
    return gamma_rain_accum


def compute_image_signal(J, A, R, gamma_rain, gamma_fog, gamma_rain_accum, d):
    """
    计算图像信号。

    参数:
    J - 背景信号 (2D array)
    A - 大气光 (scalar)
    R - 雨纹信号 (2D array)
    gamma_rain - 雨的散射系数 (scalar)
    gamma_fog - 雾的散射系数 (scalar)
    gamma_rain_accum - 雨积聚的散射系数分布 (2D array)
    d - 成像距离 (2D array)

    返回:
    I - 修正后的图像信号 (2D array)
    """
    # 雨和雨积聚对背景信号的附加散射
    gamma_total = gamma_rain + gamma_rain_accum + gamma_fog
    term1 = J * np.exp(-gamma_total * d)

    # 大气光的贡献
    term2 = A * (1 - np.exp(-gamma_total * d))

    # 雨纹信号的散射影响
    term3 = np.exp(-gamma_fog * d) * R

    # 综合信号
    I = term1 + term2 + term3
    return I


def visualize_simulation_rgb(J, A, R, gamma_rain, gamma_fog, gamma_rain_accum, d):
    """
    Visualize the intermediate and final components of the simulation for RGB images.
    """
    # 计算图像信号
    # I = compute_image_signal(J, A, R, gamma_rain, gamma_fog, gamma_rain_accum, d)

    # Visualization
    fig, axes = plt.subplots(3, 2, figsize=(20, 10))

    # Original background signal
    axes[0, 0].imshow(to_visualize(J))
    axes[0, 0].set_title('Background Signal ($J(x, y)$)')
    axes[0, 0].axis('off')

    # Rain streak signal
    scatter_R = R * np.exp(-gamma_fog * d)
    axes[0, 1].imshow(to_visualize(scatter_R))
    axes[0, 1].set_title('Rain Streak Signal ($R(x, y)$)')
    axes[0, 1].axis('off')

    # Rain streak
    rain_streak = J + scatter_R
    rain_streak = np.clip(rain_streak, 0, 1)
    axes[1, 0].imshow(to_visualize(rain_streak))
    axes[1, 0].set_title('With Rain Streak')
    axes[1, 0].axis('off')

    # Fog
    fog = J * np.exp(-gamma_fog * d) + A * (1 - np.exp(-gamma_fog * d))
    fog = np.clip(fog, 0, 1)
    axes[1, 1].imshow(to_visualize(fog))
    axes[1, 1].set_title('With Fog')
    axes[1, 1].axis('off')

    # Rain & Fog
    gamma = gamma_rain + gamma_fog
    rain_fog = J * np.exp(-gamma * d) + A * (1 - np.exp(-gamma * d))
    rain_fog = np.clip(rain_fog, 0, 1)
    axes[2, 0].imshow(to_visualize(rain_fog))
    axes[2, 0].set_title('With Fog and Rain')
    axes[2, 0].axis('off')

    # Final simulated image
    I = rain_fog + R * np.exp(-gamma_fog * d)
    I = np.clip(I, 0, 1)
    axes[2, 1].imshow(to_visualize(I))
    axes[2, 1].set_title('Final Simulated Image ($I(x, y)$)')
    axes[2, 1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 示例参数
    image_shape = (512, 512, 3)

    J = read_img('demo/BSD300/2092.jpg')
    J = resize_image(J, new_width=512, new_height=512)

    A = np.ones((512, 512, 3)) * 0.8

    R = read_img('demo/Rain_Streaks_in_high_altitude.jpg')  # 雨纹信号

    gamma_rain = np.ones((512, 512, 3)) * 0.05
    gamma_fog = np.ones((512, 512, 3)) * 0.5

    d = np.full(J.shape, 1.0)  # 成像距离

    centers = [(256, 256), (128, 128)]  # 雨积聚中心点
    sigma = 5  # 雨积聚区域的扩展范围
    alpha = 0.3  # 雨积聚的散射强度比例因子

    # 生成雨积聚散射分布
    gamma_rain_accum = generate_rain_accumulation_mask(image_shape, centers, sigma, alpha)

    visualize_simulation_rgb(J, A, R, gamma_rain, gamma_fog, gamma_rain_accum, d)
