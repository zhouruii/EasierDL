import cv2
import numpy as np
from matplotlib import pyplot as plt
from pystackreg import StackReg
from skimage import io
from skimage.transform import resize
import itk
import SimpleITK as sitk

from utils import normalize


def feature_based_registration(fixed_npy, moving_npy):
    """
    基于特征点的图像配准。

    Args:
        fixed_npy (np.ndarray): 固定图像（参考图像），形状为 (H, W)。
        moving_npy (np.ndarray): 移动图像（待配准图像），形状与固定图像一致。
        output_dir (str): 输出结果保存目录。
    """

    # 转换为 OpenCV 格式（仅支持 2D）
    fixed = fixed_npy.astype(np.uint8)
    moving = moving_npy.astype(np.uint8)

    # 初始化 SIFT 检测器
    sift = cv2.SIFT_create()

    # 检测特征点和描述符
    kp1, des1 = sift.detectAndCompute(fixed, None)
    kp2, des2 = sift.detectAndCompute(moving, None)

    # 使用 FLANN 匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 匹配特征点
    matches = flann.knnMatch(des1, des2, k=2)

    # 筛选匹配点（Lowe's ratio test）
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # 至少需要 4 对匹配点
    if len(good) < 4:
        raise ValueError("匹配点不足，无法计算变换矩阵")

    # 提取匹配点坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # 计算仿射变换矩阵
    M, _ = cv2.estimateAffine2D(src_pts, dst_pts)

    # 应用变换
    registered = cv2.warpAffine(moving, M, (fixed.shape[1], fixed.shape[0]))

    return registered


def visualize(fix, move, reg):
    fix = normalize(np.clip(fix, 0, 1))
    reg = normalize(np.clip(reg, 0, 1))
    composite = np.zeros((fix.shape[0], fix.shape[1], 3), dtype=np.float32)

    # 固定图像放到红色通道，配准图像放到绿色通道
    composite[..., 0] = fix  # 红色
    composite[..., 1] = reg  # 绿色
    io.imsave(f"reg.jpg", (composite * 255).astype(np.uint8))

    # """生成验证对比图"""
    # plt.figure(figsize=(15, 5))
    #
    # # 显示参考图像
    # plt.imshow(fix, cmap='gray', alpha=1, label='Fixed Image')
    #
    # # 显示配准后的图像
    # plt.imshow(reg, cmap='gray', alpha=1, label='Registered Image')
    #
    # plt.axis('off')
    # plt.savefig('reg.jpg', dpi=600)
    # plt.close()


if __name__ == '__main__':
    fix = np.load('/home/disk1/ZR/datasets/OurHSI/4/raw_0_rd_rf_or.npy')
    fix[fix < 0] = 0
    move = np.load('/home/disk1/ZR/datasets/OurHSI/9/raw_0_rd_rf_or.npy')
    move[move < 0] = 0
    reg = feature_based_registration(fix[19, :, :], move[19, :, :])
    visualize(fix[19, :, :], move[19, :, :], reg)
