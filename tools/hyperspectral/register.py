import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from skimage.registration import phase_cross_correlation
from skimage.transform import warp, AffineTransform
from tqdm import tqdm

from register_core import register_hyper_bands
from utils import normalize


# ========================
# 修改后的数据读取模块
# ========================
def load_envi_data(raw_path):
    """使用rasterio读取ENVI格式高光谱数据"""
    with rasterio.open(raw_path, driver='ENVI') as src:
        # 读取全部波段数据，形状为 (bands, height, width)
        data = src.read()
        # 转换为 (bands, height, width) 的numpy数组
        return data


def get_sorted_files(folder):
    """获取按编号排序的文件列表"""
    files = []
    for f in os.listdir(folder):
        if f.endswith(".hdr"):
            base = f[:-4]
            raw_path = os.path.join(folder, base)
            if os.path.exists(raw_path):
                # 提取编号（假设文件名格式：raw_数字_rd_rf_or）
                num = int(base.split('_')[1])
                files.append((num, raw_path, os.path.join(folder, f)))
    # 按编号排序
    return sorted(files, key=lambda x: x[0])


# ========================
# 可视化与保存模块
# ========================
def save_visualization(ref_band, mov_band, reg_band, save_path):
    """生成验证对比图"""
    plt.figure(figsize=(15, 5))

    # 显示参考图像
    plt.imshow(normalize(np.clip(ref_band, 0, 1)), cmap='gray', alpha=0.7, label='Fixed Image')

    # 显示配准后的图像
    plt.imshow(normalize(np.clip(reg_band, 0, 1)), cmap='gray', alpha=0.5, label='Registered Image')

    plt.axis('off')
    plt.savefig(save_path, dpi=600)
    plt.close()


# ========================
# 主程序
# ========================
def process_folders(folder1, folder2, output_folder):
    """主处理流程"""
    # 准备输出目录
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "visualization"), exist_ok=True)

    # 获取排序后的文件列表
    files1 = get_sorted_files(folder1)
    files2 = get_sorted_files(folder2)

    # 检查文件数量一致性
    assert len(files1) == len(files2), "文件夹包含文件数量不一致"

    # 处理每个文件对
    for (n1, r1, h1), (n2, r2, h2) in zip(tqdm(files1, desc="处理400-900nm数据"),
                                          files2):
        # 加载数据
        data1 = load_envi_data(r1)  # 400-900nm
        data2 = load_envi_data(r2)  # 900-2500nm

        # 执行配准
        registered = register_hyper_bands(data1, data2)

        # 合并光谱范围
        merged = np.concatenate([data1, registered], axis=0)

        # 保存数据
        np.save(os.path.join(output_folder, f"merged_{n1}_{n2}.npy"), merged)

        # 保存可视化
        save_visualization(
            data1[20], data2[20], registered[20],
            os.path.join(output_folder, "visualization", f"compare_{n1}_{n2}.png")
        )


# ========================
# h5py保存示例（可选）
# ========================
def save_h5_example(data, path):
    """HDF5保存示例"""
    import h5py
    with h5py.File(path, 'w') as f:
        dset = f.create_dataset("hyper", data=data)
        dset.attrs['description'] = "Merged hyperspectral data"
        dset.attrs['wavelength_range'] = "400-2500nm"


if __name__ == "__main__":
    # 输入路径设置
    folder_400_900 = "/home/disk1/ZR/datasets/OurHSI/4"
    folder_900_2500 = "/home/disk1/ZR/datasets/OurHSI/9"
    output_dir = "/home/disk2/ZR/datasets"

    # 执行处理
    process_folders(folder_400_900, folder_900_2500, output_dir)
