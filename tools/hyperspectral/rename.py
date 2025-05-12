import os
import shutil
import json
from pathlib import Path


def rename_npy_files(directory: str):
    """
    重命名目录下的.npy文件为连续数字编号，并保存映射关系

    Args:
        directory: 包含.npy文件的目录路径
    """
    # 转换为Path对象便于处理
    dir_path = Path(directory)

    # 验证目录是否存在
    if not dir_path.is_dir():
        raise ValueError(f"目录不存在: {directory}")

    # 获取所有.npy文件（按文件名排序保证可重复性）
    npy_files = sorted(f for f in dir_path.glob('*.npy') if f.is_file())

    if not npy_files:
        print(f"警告: 目录中没有找到.npy文件")
        return

    # 创建映射字典
    name_mapping = {}

    # 重命名文件
    for idx, file in enumerate(npy_files, start=1):
        original_name = file.stem  # 不带扩展名的原始文件名
        new_name = f"{idx}.npy"
        new_path = file.with_name(new_name)

        # 避免覆盖已有文件
        if new_path.exists():
            raise RuntimeError(f"冲突: 目标文件已存在 {new_path}")

        file.rename(new_path)
        name_mapping[original_name] = new_name

    # 保存映射表
    mapping_file = dir_path / "filename_mapping.json"
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(name_mapping, f, indent=2, ensure_ascii=False)

    print(f"完成: 重命名了 {len(npy_files)} 个文件")
    print(f"映射表已保存到: {mapping_file}")


def restore_names(directory):
    mapping_file = '/home/disk2/ZR/datasets/AVIRIS/512/mapping.json'
    with open(mapping_file) as f:
        mapping = json.load(f)

    reverse_mapping = {v: k for k, v in mapping.items()}

    for file in Path(directory).glob('*.npy'):
        if file.name in reverse_mapping:
            original_name = reverse_mapping[file.name] + '.npy'
            file.rename(file.with_name(original_name))


def remove_last_underscore_section(directory):
    """
    重命名目录下的文件：删除文件名中最后一个下划线及其后面的部分

    例如：
    输入：abc_123_xyz.npy → 输出：abc_123.npy
    输入：file_part1_part2.txt → 输出：file_part1.txt
    """
    dir_path = Path(directory)

    # 验证目录是否存在
    if not dir_path.is_dir():
        raise ValueError(f"目录不存在: {directory}")

    renamed_count = 0

    for filepath in dir_path.iterdir():
        if filepath.is_file():
            # 分离文件名和扩展名
            stem = filepath.stem
            suffix = filepath.suffix

            # 找到最后一个下划线的位置
            last_underscore = stem.rfind('_')
            if last_underscore == -1:
                continue  # 没有下划线则跳过

            # 构建新文件名
            new_stem = stem[:last_underscore]
            new_filename = f"{new_stem}{suffix}"
            new_path = filepath.with_name(new_filename)

            # 避免覆盖已有文件
            if new_path.exists():
                print(f"警告: 跳过 {filepath.name} → {new_filename} (目标文件已存在)")
                continue

            # 执行重命名
            filepath.rename(new_path)
            renamed_count += 1
            print(f"已重命名: {filepath.name} → {new_filename}")

    print(f"\n完成: 共处理了 {renamed_count} 个文件")


if __name__ == "__main__":
    target_dir = '/home/disk2/ZR/datasets/OurHSI/rain/storm'

    # rename_npy_files(target_dir)
    # restore_names(target_dir)
    remove_last_underscore_section(target_dir)
