from torch.utils.data import DataLoader, Dataset
import torch
import h5py
import numpy as np
import glob
import scipy.io
import tifffile as tiff
from PIL import Image

from .builder import DATASET


# ================= 波段得分与交换函数 ===================
def compute_scores(band_data, wavelengths, a=0.01, b=2):
    # band_data: [B, C, H, W], wavelengths: [C]
    spatial_std = band_data.std(dim=[2, 3])  # [B, C]
    wave_term = a * torch.exp(-b * (wavelengths.unsqueeze(0) - 400) / (2500 - 400))  # [1, C]
    scores = spatial_std + wave_term.to(spatial_std.device)  # [B, C]
    return scores


def get_topk_indices(scores, ratio=0.1, largest=True):
    k = max(1, int(scores.size(1) * ratio))
    values, indices = torch.topk(scores, k=k, dim=1, largest=largest)
    return indices


def exchange_band_values(group_a, group_b, indices_a, indices_b):
    group_a_new = group_a.clone()
    group_b_new = group_b.clone()
    for b in range(group_a.size(0)):
        a_idx = indices_a[b]
        b_idx = indices_b[b]
        temp = group_a[b, a_idx].clone()
        group_a_new[b, a_idx] = group_b[b, b_idx]
        group_b_new[b, b_idx] = temp
    return group_a_new, group_b_new


def compute_band_exchange_indices(hsi, first_branch_channel=102, a=0.01, b=2):
    hsi = hsi.unsqueeze(0)
    C = hsi.shape[1]
    f = first_branch_channel
    device = hsi.device

    group1 = hsi[:, 0:f]
    group2 = hsi[:, f:2 * f]
    group3 = hsi[:, 2 * f:]

    wavelengths = torch.linspace(400, 2500, C, device=device)
    w1 = wavelengths[0:f]
    w2 = wavelengths[f:2 * f]
    w3 = wavelengths[2 * f:]

    s1 = compute_scores(group1, w1, a, b)
    s2 = compute_scores(group2, w2, a, b)
    s3 = compute_scores(group3, w3, a, b)

    top1 = get_topk_indices(s1, 0.1, largest=True)
    low2 = get_topk_indices(s2, 0.1, largest=False)
    top2 = get_topk_indices(s2, 0.1, largest=True)
    low3 = get_topk_indices(s3, 0.1, largest=False)

    return top1, low2, top2, low3


def apply_band_exchange_with_indices(hsi, indices, first_branch_channel=102):
    hsi = hsi.unsqueeze(0)
    f = first_branch_channel

    group1 = hsi[:, 0:f]
    group2 = hsi[:, f:2 * f]
    group3 = hsi[:, 2 * f:]

    top1, low2, top2, low3 = indices

    group1, group2 = exchange_band_values(group1, group2, top1, low2)
    group2, group3 = exchange_band_values(group2, group3, top2, low3)

    out = torch.cat([group1, group2, group3], dim=1)
    return out.squeeze(0)


# ================= 数据集定义 ===================
def default_loader(path):
    return tiff.imread(path)


@DATASET.register_module()
class HSIDehazeDataset(Dataset):
    def __init__(self, im_list_x, im_list_y, loader=default_loader, exchange_bands=False, first_branch_channel=102):
        super(HSIDehazeDataset, self).__init__()
        self.loader = loader
        self.imlist_x = im_list_x
        self.imlist_y = im_list_y
        self.exchange_bands = exchange_bands
        self.first_branch_channel = first_branch_channel

    def __getitem__(self, index):
        im_data = self.loader(self.imlist_x[index])
        str = self.imlist_x[index]
        str = str.replace("haze", "clean")
        str = list(str)
        len1 = len(str)
        if str[len1 - 7] == "_":
            str[len1 - 7:len1 - 4] = ""
        else:
            str[len1 - 6:len1 - 4] = ""
        str = "".join(str)

        im_data = np.asanyarray(im_data, dtype="float32") / 2200
        im_data = torch.Tensor(im_data).permute(2, 0, 1)

        im_label = self.loader(str)
        im_label = np.asanyarray(im_label, dtype="float32") / 2200
        im_label = torch.Tensor(im_label)

        if self.exchange_bands:
            indices = compute_band_exchange_indices(im_data, first_branch_channel=self.first_branch_channel)
            im_data = apply_band_exchange_with_indices(im_data, indices, first_branch_channel=self.first_branch_channel)
            im_label = apply_band_exchange_with_indices(im_label, indices,
                                                        first_branch_channel=self.first_branch_channel)

        return im_data, im_label

    def __len__(self):
        return len(self.imlist_x)
