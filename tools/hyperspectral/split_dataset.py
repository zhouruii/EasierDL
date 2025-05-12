import os
import shutil
import random
from pathlib import Path


def split_dataset(root_dir: str, ratios: tuple = (0.7, 0.2, 0.1), seed: int = 42):
    """
    分割数据集到train/val/test目录

    Args:
        root_dir: 原始数据根目录（包含gt和rain）
        ratios: 分割比例（train, val, test）
        seed: 随机种子
    """
    # 验证输入
    assert sum(ratios) == 1.0, "比例总和必须为1"
    assert len(ratios) == 3, "需要3个比例值"

    # 设置随机种子
    random.seed(seed)

    # 转换为Path对象
    src_dir = Path(root_dir)
    dest_dirs = {
        'train': src_dir / 'train',
        'val': src_dir / 'val',
        'test': src_dir / 'test'
    }

    # 验证源目录结构
    gt_dir = src_dir / 'gt'
    rain_dir = src_dir / 'rain'
    rain_subdirs = ['small', 'medium', 'heavy', 'storm']

    if not gt_dir.exists():
        raise ValueError(f"缺少gt目录: {gt_dir}")
    if not rain_dir.exists():
        raise ValueError(f"缺少rain目录: {rain_dir}")

    # 获取所有基础文件名（不带扩展名）
    gt_files = [f.stem for f in gt_dir.glob('*.npy')]
    if not gt_files:
        raise ValueError(f"gt目录中没有找到.npy文件")

    # 检查rain子目录中的文件是否匹配
    for sub in rain_subdirs:
        rain_files = {f.stem for f in (rain_dir / sub).glob('*.npy')}
        missing = set(gt_files) - rain_files
        if missing:
            raise ValueError(f"rain/{sub}中缺少与gt匹配的文件: {missing}")

    # 打乱文件列表（保持一致性）
    random.shuffle(gt_files)

    # 计算分割点
    total = len(gt_files)
    train_end = int(total * ratios[0])
    val_end = train_end + int(total * ratios[1])

    splits = {
        'train': gt_files[:train_end],
        'val': gt_files[train_end:val_end],
        'test': gt_files[val_end:]
    }

    # 创建目标目录结构
    for split_name in dest_dirs:
        (dest_dirs[split_name] / 'gt').mkdir(parents=True, exist_ok=True)
        for sub in rain_subdirs:
            (dest_dirs[split_name] / 'rain' / sub).mkdir(parents=True, exist_ok=True)

    # 执行文件复制
    for split_name, files in splits.items():
        print(f"处理 {split_name} 集 ({len(files)} 个文件)...")

        for fname in files:
            # 复制gt文件
            src = gt_dir / f"{fname}.npy"
            dst = dest_dirs[split_name] / 'gt' / f"{fname}.npy"
            shutil.copy2(src, dst)

            # 复制rain子目录文件
            for sub in rain_subdirs:
                src = rain_dir / sub / f"{fname}.npy"
                dst = dest_dirs[split_name] / 'rain' / sub / f"{fname}.npy"
                shutil.copy2(src, dst)

    print("\n分割完成！")
    print(f"train: {len(splits['train'])} 文件")
    print(f"val: {len(splits['val'])} 文件")
    print(f"test: {len(splits['test'])} 文件")


if __name__ == "__main__":

    root_dir = '/home/disk2/ZR/datasets/OurHSI/extra'
    ratios = (0.8, 0.2, 0.0)
    seed = 42

    split_dataset(
        root_dir=root_dir,
        ratios=ratios,
        seed=seed
    )
