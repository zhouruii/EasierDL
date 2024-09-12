import argparse
import os
import shutil
import random


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description='Splitting images')
    parser.add_argument("--train", type=str, default='data/spectral/train/reflectivity', )
    parser.add_argument("--val", type=str, default='data/spectral/val/reflectivity', )
    parser.add_argument("--nproc", type=int, default=10)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # 设置随机种子，以便每次运行都能得到相同的划分结果
    random.seed(42)

    # 源目录和目标目录
    train_dir = args.train
    val_dir = args.val

    # 创建目标目录，如果不存在则创建
    os.makedirs(val_dir, exist_ok=True)

    # 获取所有图片文件列表
    all_images = os.listdir(train_dir)
    all_images = sorted(all_images, key=lambda x: int(x.split('.')[0]))

    # 设置验证集的比例，例如 20%
    val_ratio = 0.3

    # 计算验证集的大小
    val_size = int(len(all_images) * val_ratio)

    # 随机选择验证集的图片
    val_images = random.sample(all_images, val_size)

    # 复制图片到验证集目录
    for image in val_images:
        src_path = os.path.join(train_dir, image)
        dst_path = os.path.join(val_dir, image)
        shutil.move(src_path, dst_path)
        print(f'Move {src_path} images to {dst_path}')


if __name__ == '__main__':
    main()
