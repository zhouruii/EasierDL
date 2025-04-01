import pickle

import numpy as np


def read_pkl(file_path):
    """读取 .pkl 文件。

    Args:
        file_path (str): .pkl 文件路径。

    Returns:
        dict: 从 .pkl 文件中读取的数据。
    """
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            print(f"成功读取文件: {file_path}")
            return data
    except FileNotFoundError:
        print(f"文件不存在: {file_path}")
        return {}
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        return {}


def write_pkl(file_path, data):
    """将数据保存到 .pkl 文件。

    Args:
        file_path (str): .pkl 文件路径。
        data (dict): 需要保存的数据。
    """
    try:
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
            print(f"成功保存文件: {file_path}")
    except Exception as e:
        print(f"保存文件时出错: {str(e)}")


def modify_data(data, key, value):
    """修改或添加数据。

    Args:
        data (dict): 需要修改的数据。
        key (str): 键。
        value (any): 值。

    Returns:
        dict: 修改后的数据。
    """
    data[key] = value
    print(f"已修改或添加数据: {key} = {value}")
    return data


# 示例使用
if __name__ == "__main__":
    # 文件路径
    pkl_file = "/home/disk2/ZR/datasets/OurHSI/meta.pkl"

    # 读取 .pkl 文件
    data = read_pkl(pkl_file)

    first_remove_bands = data['bands']
    bands = np.delete(first_remove_bands, [465], axis=0)

    # 修改或添加数据
    data = modify_data(data, "NaN_band after first remove", [465])
    data = modify_data(data, "first_remove_bands", first_remove_bands)
    data = modify_data(data, "bands", bands)

    # 保存修改后的数据
    write_pkl(pkl_file, data)
