from os import listdir
from os.path import join

import torch


def read_pts(data_root, is1d=False):
    name_list = listdir(data_root)
    name_list = sorted(name_list, key=lambda x: int(x.split('.')[0]))
    path_list = [join(data_root, name) for name in name_list]
    if is1d:
        data = [torch.load(path) for path in path_list]
    else:
        data = [torch.load(path).permute(2, 0, 1) for path in path_list]
    return data

