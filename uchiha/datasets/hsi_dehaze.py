from typing import List, Dict
import numpy as np
import glob
from scipy import io
import tifffile as tiff
from functools import partial
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from torch.utils.data import Dataset
import torch

from .builder import DATASET
from .pipelines import Compose
from ..utils import get_root_logger


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


def get_clean_of_HD(haze_path):
    clean_path = haze_path.replace("haze", "clean")
    clean_path = list(clean_path)
    len1 = len(clean_path)
    if clean_path[len1 - 7] == "_":
        clean_path[len1 - 7:len1 - 4] = ""
    else:
        clean_path[len1 - 6:len1 - 4] = ""
    clean_path = "".join(clean_path)
    return clean_path


# ================= 数据集定义 ===================
def get_loader(loader_type, path):
    if loader_type == 'tiff':
        return tiff.imread(path)
    elif loader_type == 'mat':
        return io.loadmat(path)
    else:
        raise ValueError('Invalid loader type')


def load_hd(loader, path, exchange_bands=False, first_branch_channel=102):
    im_data = loader(path)
    clean_path = get_clean_of_HD(path)

    im_data = np.asanyarray(im_data, dtype="float32") / 2200
    im_data = torch.Tensor(im_data).permute(2, 0, 1)

    im_label = loader(clean_path)
    im_label = np.asanyarray(im_label, dtype="float32") / 2200
    im_label = torch.Tensor(im_label)

    if exchange_bands:
        indices = compute_band_exchange_indices(im_data, first_branch_channel=first_branch_channel)
        im_data = apply_band_exchange_with_indices(im_data, indices, first_branch_channel=first_branch_channel)
        im_label = apply_band_exchange_with_indices(im_label, indices,
                                                    first_branch_channel=first_branch_channel)
    return im_data, im_label


def load_mat(loader, gt_path, lq_path):
    im_data = loader(lq_path)
    data_key = list(im_data.keys())[3]
    im_data = im_data[data_key]
    im_data = np.array(im_data)
    im_data = torch.Tensor(im_data)

    im_label = loader(gt_path)
    label_key = list(im_label.keys())[3]
    im_label = im_label[label_key]
    im_label = np.array(im_label)
    im_label = torch.Tensor(im_label)

    #######
    im_data = im_data.permute(2, 0, 1)
    im_label = im_label.permute(2, 0, 1)
    return im_data, im_label


@DATASET.register_module()
class HSIDehazeDataset(Dataset):
    def __init__(self, gt_path=None, lq_path=None, loader_type='tiff', dataset_name='HD', pipelines=None):
        super(HSIDehazeDataset, self).__init__()
        self.loader = partial(get_loader, loader_type)
        self.gt_path = glob.glob(gt_path)
        self.lq_path = glob.glob(lq_path)
        self.dataset_name = dataset_name
        self.pipelines = Compose(pipelines) if pipelines else None

    def __getitem__(self, index):
        if self.dataset_name == 'HD':
            lq, gt = load_hd(self.loader, self.lq_path[index])
        elif self.dataset_name == 'AVIRIS':
            lq, gt = load_mat(self.loader, self.gt_path[index], self.lq_path[index])
        elif self.dataset_name == 'UAV':
            lq, gt = load_mat(self.loader, self.gt_path[index], self.lq_path[index])
        else:
            raise ValueError('Invalid dataset name')

        results = {
            'sample': lq,
            'target': gt,
            'index': index,
        }

        return self.pipelines(results) if self.pipelines else results

    def __len__(self):
        return len(self.gt_path)

    def evaluate(self, preds: List[np.ndarray], targets: List[np.ndarray], metric,
                 indexes: List[int]) -> dict:
        """
        同时计算PSNR和SSIM指标

        Args:
            preds: 模型输出列表，每个元素形状为 (B, C, H, W)
            targets: 真实标签列表，每个元素形状为 (B, C, H, W)
            metric: 需要计算的指标，默认为['PSNR','SSIM']
            indexes: 原始数据索引列表

        Returns:
            dict: 包含平均PSNR和SSIM的字典
        """
        assert len(preds) == len(targets) == len(indexes), "输入列表长度必须一致"
        assert metric is None or metric.upper() == 'PSNR' or metric.upper() == 'SSIM', \
            '默认只支持PSNR和SSIM'

        logger = get_root_logger()
        logger.info('start evaluating...')

        psnrs = []
        ssims = []
        for pred_batch, target_batch, index_batch in zip(preds, targets, indexes):
            B, C, H, W = pred_batch.shape

            for i in range(B):
                # 获取当前样本
                pred = pred_batch[i]  # (C,H,W)
                target = target_batch[i]

                # 转换为(H,W,C)格式
                if C == 1:
                    pred_img = pred[0]
                    target_img = target[0]
                else:
                    pred_img = np.transpose(pred, (1, 2, 0))
                    target_img = np.transpose(target, (1, 2, 0))

                # 计算PSNR
                data_range = target_img.max() - target_img.min()
                psnr_val = psnr(target_img, pred_img, data_range=data_range)

                # 计算SSIM
                multichannel = C > 1
                ssim_val = ssim(
                    target_img, pred_img,
                    multichannel=multichannel,
                    channel_axis=2 if multichannel else None,
                    data_range=data_range
                )

                psnrs.append(psnr_val)
                ssims.append(ssim_val)

        mean_psnr = np.mean(psnrs)
        mean_ssim = np.mean(ssims)
        logger.info(f"Mean PSNR: {mean_psnr:.2f} dB")
        logger.info(f"Mean SSIM: {mean_ssim:.4f}")

        return {
            'psnr': mean_psnr,
            'ssim': mean_ssim
        }
