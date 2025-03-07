import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from tools.synthesis.config import RAIN_STREAK, RAIN_STREAK_V2
from tools.synthesis.rain_3d import generate_3d_rain


def generate_raindrops(num_drops, width, height, min_depth, max_depth):
    # 生成雨滴的随机三维坐标
    x = np.random.uniform(-width / 2, width / 2, num_drops)
    y = np.random.uniform(-height / 2, height / 2, num_drops)
    z = np.random.uniform(min_depth, max_depth, num_drops)
    return np.column_stack((x, y, z))


def project_perspective(drops, fov_deg, speed_x, speed_y, width, height):
    # 计算透视投影比例（基于Z值）
    fov_rad = np.deg2rad(fov_deg)
    scale = 0.5 * width / np.tan(fov_rad / 2)
    # scale = 0.5 * width * math.sqrt(2) / np.tan(fov_rad / 2)

    # 投影到2D屏幕坐标（考虑动态模糊）
    x_proj = (drops[:, 0] * scale / drops[:, 2]) + width / 2
    y_proj = (drops[:, 1] * scale / drops[:, 2]) + height / 2

    # 根据速度生成拖影（运动模糊）
    blur_length = np.sqrt(speed_x ** 2 + speed_y ** 2) * 0.1
    steps = int(blur_length) + 1
    offsets_x = np.linspace(0, speed_x, steps)
    offsets_y = np.linspace(0, speed_y, steps)

    # 生成拖影轨迹点
    trajectory = []
    for dx, dy in zip(offsets_x, offsets_y):
        x_traj = ((drops[:, 0] + dx) * scale / drops[:, 2]) + width / 2
        y_traj = ((drops[:, 1] + dy) * scale / drops[:, 2]) + height / 2
        trajectory.append(np.column_stack((x_traj, y_traj, drops[:, 2])))

    return np.concatenate(trajectory)


def render_rain(projected_points, width, height, drop_size_base):
    z = projected_points[:, 2]
    # 创建空白图像
    image = np.zeros((height, width), dtype=np.float32)

    # 根据深度调整雨滴大小和透明度
    # sizes = drop_size_base * (MAX_DEPTH / projected_points[:, 2])
    # alphas = np.clip(1.0 / (projected_points[:, 2] * 0.1), 0.1, 1.0)
    sizes = drop_size_base * (z / z.max())
    alphas = np.clip(z / z.max(), 0.1, 1.0)

    # 绘制雨滴（带大小和透明度）
    for (x, y, _), size, alpha in zip(projected_points, sizes, alphas):
        if 0 <= x < width and 0 <= y < height:
            radius = int(size)
            cv2.circle(image, (int(x), int(y)), radius, alpha, -1)

    # 高斯模糊增强雨纹效果
    image = gaussian_filter(image, sigma=1)
    image = (image / image.max()) * 255
    return image.astype(np.uint8)


if __name__ == '__main__':
    # 图像尺寸
    WIDTH, HEIGHT = 256, 256

    # 雨滴参数
    NUM_DROPS = 2000  # 雨滴数量
    MIN_DEPTH = 1.0  # 最近距离
    MAX_DEPTH = 45.0  # 最远距离
    DROP_SIZE_BASE = 1  # 基础大小（近处雨滴）
    SPEED_Y = 1.0  # 下落速度（Y方向，俯视视角下为纵向）
    SPEED_X = 1.5  # 横向风速（X方向）
    FOV = 60  # 视角（度数，控制透视变形强度）

    # 生成初始雨滴
    x, y, z = generate_3d_rain(height=HEIGHT, width=WIDTH, depth=256, **RAIN_STREAK_V2[4])
    z = np.array(z)
    z[z < 1] = 1
    # Center the points
    x_center, y_center = WIDTH // 2, HEIGHT // 2  # x.mean(), y.mean()
    x = np.array(x) - x_center
    y = np.array(y) - y_center

    points = np.vstack((x, y, z)).T

    # drops = generate_raindrops(NUM_DROPS, WIDTH, HEIGHT, MIN_DEPTH, MAX_DEPTH)
    drops = points

    # 执行投影（生成拖影轨迹点）
    projected_points = project_perspective(drops, FOV, SPEED_X, SPEED_Y, WIDTH, HEIGHT)

    # 渲染俯视图
    top_view = render_rain(projected_points, WIDTH, HEIGHT, DROP_SIZE_BASE)

    # 俯视图（核心）
    plt.figure(figsize=(6, 6))
    plt.imshow(top_view, cmap='gray')
    plt.title("Top View")
    plt.axis('off')

    # 正视图（侧向，可选）
    # 修改投影方向参数即可生成其他视图
    # ...

    plt.show()
