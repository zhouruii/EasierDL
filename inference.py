import argparse
import glob
import os
from collections import OrderedDict
from typing import Tuple

import numpy as np
import tifffile
from matplotlib import pyplot as plt
from scipy import io
import torch

from uchiha.apis import set_random_seed
from uchiha.apis.inference import simple_inference_with_h5
from uchiha.datasets.builder import build_dataset, build_dataloader
from uchiha.models.builder import build_model
from uchiha.utils import load_config


def remove_module_prefix(state_dict):
    """移除权重中的module.前缀"""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_key = k[7:]
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict


def load_data(data_path):
    ext = os.path.splitext(data_path)[1].lower()
    if ext == '.npy':
        data = np.load(data_path)
    elif ext in ('.tif', '.tiff'):
        data = tifffile.imread(data_path)
        if data.shape[0] != 512:
            data = np.transpose(data, (1, 2, 0))
        data = data[:, :, :305]
        data = np.asanyarray(data, dtype="float32") / 2900
    elif ext == '.mat':
        full_data = io.loadmat(data_path)
        data_key = list(full_data.keys())[3]
        data = full_data[data_key]
    else:
        raise ValueError(f"不支持的格式: {ext}")

    return data


def crop_and_pad(img: np.ndarray, crop_size: int, overlap: int) -> Tuple[list, Tuple[int, int]]:
    """
    将图像分割成重叠的裁剪块

    参数:
        img: 输入图像 (H,W,C)
        crop_size: 裁剪尺寸
        overlap: 重叠区域大小

    返回:
        (裁剪块列表, 原始图像填充后的形状)
    """
    h, w, c = img.shape
    stride = crop_size - overlap

    # 计算需要的填充量
    pad_h = (stride - h % stride) % stride
    pad_w = (stride - w % stride) % stride

    # 对称填充
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # 使用边缘反射填充
    img_padded = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                        mode='reflect')

    # 生成裁剪块
    patches = []
    new_h, new_w = img_padded.shape[:2]

    for y in range(0, new_h - overlap, stride):
        for x in range(0, new_w - overlap, stride):
            patch = img_padded[y:y + crop_size, x:x + crop_size, :]
            patches.append(((y, x), patch))

    return patches, (pad_top, pad_left)


def merge_patches(patches: list, original_shape: Tuple[int, int, int],
                  crop_size: int, overlap: int) -> np.ndarray:
    """
    合并重叠的裁剪块

    参数:
        patches: 包含位置和预测结果的裁剪块列表
        original_shape: 原始图像形状 (H,W,C)
        crop_size: 裁剪尺寸
        overlap: 重叠区域大小

    返回:
        合并后的图像
    """
    h, w, c = original_shape
    stride = crop_size - overlap
    pad_top, pad_left = patches[0][0]  # 获取填充信息

    # 计算填充后的形状
    padded_h = ((h + stride - 1) // stride) * stride + overlap
    padded_w = ((w + stride - 1) // stride) * stride + overlap

    # 初始化输出
    output = np.zeros((padded_h, padded_w, c), dtype=np.float32)

    for (y, x), patch in patches:

        # 累加到输出图像
        output[y:y + crop_size, x:x + crop_size] = patch

    # 移除填充
    output = output[pad_top:pad_top + h, pad_left:pad_left + w]

    return output


def single_visualize(lq, pred, gt, name, output_dir, channel_indices=(0, 1, 2)):
    def extract_rgb(data):
        return np.stack([
            np.clip(data[:, :, channel_indices[0]], 0, 1),  # R
            np.clip(data[:, :, channel_indices[1]], 0, 1),  # G
            np.clip(data[:, :, channel_indices[2]], 0, 1)  # B
        ], axis=-1)

    lq_rgb = extract_rgb(lq)
    pred_rgb = extract_rgb(pred)
    gt_rgb = extract_rgb(gt)

    # 创建可视化图像
    plt.figure(figsize=(15, 5))

    # 显示低质量数据
    plt.subplot(1, 3, 1)
    plt.imshow(lq_rgb)
    plt.title(f'Low Quality\n{name}')
    plt.axis('off')

    # 显示修复结果
    plt.subplot(1, 3, 2)
    plt.imshow(pred_rgb)
    plt.title(f'Restored\n{name}')
    plt.axis('off')

    # 显示GT
    plt.subplot(1, 3, 3)
    plt.imshow(gt_rgb)
    plt.title(f'Ground Truth\n{name}')
    plt.axis('off')

    # 调整布局并保存
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'compare_{name}.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def parse_args():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--seed', type=int, default=49)
    args_parser.add_argument('--config', type=str, default='configs/hsi_dehaze/HD/inference/D3.yaml')
    args_parser.add_argument('--checkpoint', '-c',
                             default='experiments/hsi_dehaze/HD/comparison/D3/checkpoints/30.pth')
    return args_parser.parse_args()


def inference_wo_crop():
    args = parse_args()
    cfg = load_config(args.config)

    # random seed
    set_random_seed(args.seed)

    # dataset & dataloader
    testset = build_dataset(cfg.data.test.dataset.to_dict(), phase='test')
    testloader = build_dataloader(testset, cfg.data.test.dataloader.to_dict(), phase='test')

    # model
    model = build_model(cfg.model.to_dict())
    model = model.cuda()

    # resume
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(remove_module_prefix(checkpoint['state_dict']))

    # inference
    simple_inference_with_h5(dataloader=testloader,
                             model=model,
                             h5_file='/home/uchiha/datasets/inference.h5')


def get_model_and_cfg():
    args = parse_args()
    cfg = load_config(args.config)
    # random seed
    set_random_seed(args.seed)
    # model
    model = build_model(cfg.model.to_dict())
    model = model.cuda()
    # resume
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(remove_module_prefix(checkpoint['state_dict']))

    return model, cfg


def single_inference(data_path, gt_path=None):
    model, cfg = get_model_and_cfg()

    data = load_data(data_path)
    gt = load_data(gt_path)
    with torch.no_grad():
        # data
        sample = torch.from_numpy(data).permute(2, 0, 1).unsqueeze(0).cuda()
        # forward
        pred = model(sample)
        pred = pred.squeeze().permute(1, 2, 0).cpu().numpy()

    name = os.path.basename(data_path).split('.')[0]
    output_dir = cfg.work_dir
    single_visualize(lq=data, pred=pred, gt=gt, name=name, output_dir=output_dir, channel_indices=(59, 38, 20))


def single_inference_with_crop(data_path, gt_path=None, crop_size=256, overlap=0):
    model, cfg = get_model_and_cfg()

    data = load_data(data_path)
    gt = load_data(gt_path)
    # 裁剪处理
    patches, padding_info = crop_and_pad(data, crop_size, overlap)

    pred_patches = []
    with torch.no_grad():
        for pos, patch in patches:
            # 转换为模型输入格式
            sample = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).cuda()

            # 推理
            pred = model(sample)
            pred = pred.squeeze().permute(1, 2, 0).cpu().numpy()
            pred_patches.append((pos, pred))

    # 合并预测结果
    pred = merge_patches(pred_patches, data.shape, crop_size, overlap)

    # 可视化
    name = os.path.basename(data_path).split('.')[0]
    output_dir = cfg.work_dir
    single_visualize(lq=data, pred=pred, gt=gt, name=name, output_dir=output_dir,
                     channel_indices=(59, 38, 20))


def batch_inference(data_paths, gt_paths=None):
    data_paths = glob.glob(data_paths)
    gt_paths = glob.glob(gt_paths)
    for (lq, gt) in zip(data_paths, gt_paths):
        single_inference(data_path=lq, gt_path=gt)


if __name__ == '__main__':
    # HD
    # data_path = '/media/mango/系统/6636数据/2/mk/dataset/HD/test/haze/17_1.tif'
    # gt_path = '/media/mango/系统/6636数据/2/mk/dataset/HD/test/clean/17.tif'
    # single_inference(data_path, gt_path)

    # HDD
    data_path = '/media/mango/系统/6636数据/2/mk/dataset/GF5最终数据/test/haze/GF2.tif'
    gt_path = '/media/mango/系统/6636数据/2/mk/dataset/GF5最终数据/test/clean/GF2.tif'
    single_inference(data_path, gt_path)

    # data_path = '/media/mango/系统/6636数据/2/mk/dataset/GF5最终数据/test/haze/GF3.tif'
    # gt_path = '/media/mango/系统/6636数据/2/mk/dataset/GF5最终数据/test/clean/GF3.tif'
    # single_inference_with_crop(data_path, gt_path, crop_size=256, overlap=0)

    # data_paths = '/media/mango/系统/6636数据/2/mk/dataset/GF5最终数据/test/haze/*.tif'
    # gt_paths = '/media/mango/系统/6636数据/2/mk/dataset/GF5最终数据/test/clean/*.tif'
    # batch_inference(data_paths, gt_paths)
