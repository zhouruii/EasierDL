import os


def find_prefixes(directory):
    """查找目录下所有文件的 prefix。

    Args:
        directory (str): 目录路径。

    Returns:
        set: 所有文件的 prefix 集合。
    """
    # 获取所有文件的 prefix
    prefixes = set()
    for filename in os.listdir(directory):
        if "_" in filename:
            # 找到最后一个下划线的位置
            last_underscore_index = filename.rfind("_")
            prefix = filename[:last_underscore_index]  # 提取 prefix
            prefixes.add(prefix)
    return prefixes


def find_redundant_files(directory, prefix):
    """查找具有相同 prefix 的冗余文件。

    Args:
        directory (str): 目录路径。
        prefix (str): 文件名的前缀。

    Returns:
        list: 冗余文件的路径列表。
    """
    # 查找所有具有相同 prefix 的文件
    files = [f for f in os.listdir(directory) if f.startswith(prefix + "_")]

    # 如果文件数量不为 3，则没有冗余文件
    if len(files) != 3:
        return []

    # 按文件名排序
    files.sort()

    # 保留第一个文件，删除其他两个
    return [os.path.join(directory, f) for f in files[1:]]


def remove_all_redundant_files(directory):
    """删除目录下所有具有相同 prefix 的冗余文件。

    Args:
        directory (str): 目录路径。
    """
    # 查找所有 prefix
    prefixes = find_prefixes(directory)

    # 遍历每个 prefix，删除冗余文件
    for prefix in prefixes:
        redundant_files = find_redundant_files(directory, prefix)
        for file_path in redundant_files:
            os.remove(file_path)
            print(f"已删除冗余文件: {file_path}")


# 示例使用
if __name__ == "__main__":
    # 目录路径
    directory = "/home/disk2/ZR/datasets/OurHSI/extra/gt"  # 替换为你的目录路径

    # 删除所有冗余文件
    remove_all_redundant_files(directory)
