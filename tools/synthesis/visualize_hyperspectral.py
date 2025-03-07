import matplotlib.pyplot as plt
import numpy as np

# # 1. 加载高光谱图像
# file_path = r"E:\datasets\1.tif"  # 替换为你的文件路径
# with rasterio.open(file_path) as src:
#     # 读取所有波段数据
#     hyperspectral_data = src.read()  # 形状为 (波段数, 高度, 宽度)
#     print(f"图像形状 (波段数, 高度, 宽度): {hyperspectral_data.shape}")
#
#     # 查看元数据
#     print(f"波段数: {src.count}")
#     print(f"分辨率: {src.res}")
#     print(f"图像范围: {src.bounds}")

hyperspectral_data = np.load("demo.npy")
hyperspectral_data[hyperspectral_data < 0] = 0
hyperspectral_data = np.transpose(hyperspectral_data, (2, 0, 1))

# 2. 伪彩色展示
# 方法1：直接选择RGB波段（假设波段1、2、3为RGB）
if hyperspectral_data.shape[0] >= 3:
    rgb_image = hyperspectral_data[[40, 25, 10], :, :]  # 选择前3个波段
    rgb_image = np.transpose(rgb_image, (1, 2, 0))  # 转换为 (高度, 宽度, 波段数)
    rgb_image = (rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image))  # 归一化
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_image)
    plt.title('RGB Pseudo-color Image')
    plt.axis('off')
    plt.show()
else:
    print("波段数不足，无法生成RGB图像。")

# 方法2：使用PCA降维生成伪彩色图像
from sklearn.decomposition import PCA

# 将高光谱数据重塑为 (像素数, 波段数)
h, w, bands = hyperspectral_data.shape[1], hyperspectral_data.shape[2], hyperspectral_data.shape[0]
pixels = hyperspectral_data.reshape(bands, -1).T  # 形状为 (像素数, 波段数)

# 使用PCA降维到3个主成分
pca = PCA(n_components=3)
pca_result = pca.fit_transform(pixels)  # 形状为 (像素数, 3)

# 将PCA结果重塑为图像
pca_image = pca_result.reshape(h, w, 3)  # 形状为 (高度, 宽度, 3)
pca_image = (pca_image - np.min(pca_image)) / (np.max(pca_image) - np.min(pca_image))  # 归一化

# 显示PCA伪彩色图像
plt.figure(figsize=(10, 10))
plt.imshow(pca_image)
plt.title('PCA Pseudo-color Image')
plt.axis('off')
plt.show()
