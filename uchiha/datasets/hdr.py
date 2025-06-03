import glob
import os
from typing import List, Dict

import h5py
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tabulate import tabulate

from torch.utils.data import Dataset

from .builder import DATASET
from .pipelines import Compose
from ..utils import get_root_logger


@DATASET.register_module()
class HDF5MultiLevelRainHSIDataset(Dataset):
    """基于HDF5存储的多级噪声HSI数据集"""
    NOISE_LEVELS = ['small', 'medium', 'heavy', 'storm']

    def __init__(self, h5_path, split_file, pipelines=None):
        """
        Args:
            h5_path: HDF5文件路径
            split_file: 划分文件的txt路径
            pipelines: 数据预处理流水线
        """
        self.h5_path = h5_path
        self.pipelines = Compose(pipelines) if pipelines else None

        # 加载划分文件
        with open(split_file, 'r') as f:
            self.sample_names = [line.strip() for line in f if line.strip()]

        # 初始化HDF5连接（延迟到实际使用时打开）
        self._h5 = None
        self._validate_samples()

    @property
    def h5(self):
        """延迟加载HDF5文件（支持多进程）"""
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r', libver='latest', swmr=True)
        return self._h5

    def _validate_samples(self):
        """验证HDF5中是否存在所有样本"""
        # hdf5中没有后缀名
        with h5py.File(self.h5_path, 'r') as hf:
            missing = []
            for name in self.sample_names:
                name = name.replace('.npy', '')
                # 验证GT
                if f'gt/{name}' not in hf:
                    missing.append(f'gt/{name}')

                # 验证各噪声级别
                for level in self.NOISE_LEVELS:
                    if f'rain/{level}/{name}' not in hf:
                        missing.append(f'rain/{level}/{name}')

            if missing:
                raise KeyError(f"Missing {len(missing)} samples in HDF5, e.g.: {missing[:3]}")

    def __len__(self):
        return len(self.sample_names) * len(self.NOISE_LEVELS)

    def __getitem__(self, idx):
        # 计算样本名和噪声级别的对应关系
        sample_idx = idx // len(self.NOISE_LEVELS)
        noise_idx = idx % len(self.NOISE_LEVELS)

        sample_name = self.sample_names[sample_idx]
        noise_level = self.NOISE_LEVELS[noise_idx]
        sample_name = sample_name.replace('.npy', '')

        # 从HDF5加载数据（内存映射）
        gt = self.h5[f'gt/{sample_name}'][:]
        lq = self.h5[f'rain/{noise_level}/{sample_name}'][:]

        results = {
            'sample': lq,
            'target': gt,
            'index': idx,
            'noise_level': noise_level
        }

        return self.pipelines(results) if self.pipelines else results

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

        results = {
            'global': {'psnr': [], 'ssim': []},
            'by_level': {level: {'psnr': [], 'ssim': []} for level in self.NOISE_LEVELS}
        }
        logger = get_root_logger()
        logger.info('start evaluating...')

        for pred_batch, target_batch, index_batch in zip(preds, targets, indexes):
            B, C, H, W = pred_batch.shape

            for i in range(B):
                # 获取当前样本
                pred = pred_batch[i]  # (C,H,W)
                target = target_batch[i]
                idx = index_batch[i]

                # 获取噪声级别
                noise_level = self.NOISE_LEVELS[idx % len(self.NOISE_LEVELS)]

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

                # 存储结果
                results['global']['psnr'].append(psnr_val)
                results['global']['ssim'].append(ssim_val)
                results['by_level'][noise_level]['psnr'].append(psnr_val)
                results['by_level'][noise_level]['ssim'].append(ssim_val)

        # 计算统计量
        def calc_stats(values):
            return {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }

        stats = {
            'global': {
                'psnr': calc_stats(results['global']['psnr']),
                'ssim': calc_stats(results['global']['ssim'])
            },
            'by_level': {
                level: {
                    'psnr': calc_stats(results['by_level'][level]['psnr']),
                    'ssim': calc_stats(results['by_level'][level]['ssim'])
                } for level in self.NOISE_LEVELS
            }
        }

        # 日志输出
        logger.info(f"{' Evaluation Results ':=^60}")

        # 构建分层指标表格
        table_data = []
        for level in self.NOISE_LEVELS:
            table_data.append([
                level.upper(),
                f"{stats['by_level'][level]['psnr']['mean']:.2f} ± {stats['by_level'][level]['psnr']['std']:.2f}",
                f"{stats['by_level'][level]['psnr']['min']:.2f}-{stats['by_level'][level]['psnr']['max']:.2f}",
                f"{stats['by_level'][level]['ssim']['mean']:.4f} ± {stats['by_level'][level]['ssim']['std']:.4f}",
                f"{stats['by_level'][level]['ssim']['min']:.4f}-{stats['by_level'][level]['ssim']['max']:.4f}",
                stats['by_level'][level]['psnr']['count']
            ])
        table_data.append(['TOTAL',
                           f"{stats['global']['psnr']['mean']:.2f} ± {stats['global']['psnr']['std']:.2f}",
                           f"{stats['global']['psnr']['min']:.2f}-{stats['global']['psnr']['max']:.2f}",
                           f"{stats['global']['ssim']['mean']:.4f} ± {stats['global']['ssim']['std']:.4f}",
                           f"{stats['global']['ssim']['min']:.4f}-{stats['global']['ssim']['max']:.4f}",
                           stats['global']['psnr']['count']
                           ])

        # 打印表格
        logger.info("\n" + tabulate(
            table_data,
            headers=[
                "Noise Level",
                "PSNR (dB)",
                "PSNR Range",
                "SSIM",
                "SSIM Range",
                "Samples"
            ],
            tablefmt="fancy_grid",
            floatfmt=("", ".2f", "", ".4f", "", "d")
        ))
        logger.info(f"Mean PSNR: {stats['global']['psnr']['mean']:.2f} dB")
        logger.info(f"Mean SSIM: {stats['global']['ssim']['mean']:.4f}")
        logger.info("=" * 60)

        return {
            'psnr': stats['global']['psnr']['mean'],
            'ssim': stats['global']['ssim']['mean']
        }

    def get_filename(self, idx):
        """获取原始文件名（不含噪声级别信息）"""
        sample_idx = idx // len(self.NOISE_LEVELS)
        return self.sample_names[sample_idx]

    def __del__(self):
        """确保关闭HDF5连接"""
        if self._h5 is not None:
            self._h5.close()


@DATASET.register_module()
class MultiLevelRainHSIDataset(Dataset):
    # ['Light', 'Moderate', 'Heavy', 'Torrential']
    NOISE_LEVELS = ['small', 'medium', 'heavy', 'storm']

    def __init__(self, root_dir, split_file, pipelines=None):

        self.root_dir = root_dir
        self.pipelines = Compose(pipelines) if pipelines else None

        # load txt for splitting
        with open(split_file, 'r') as f:
            self.file_list = [line.strip() for line in f if line.strip()]

        # 验证文件存在性
        self._validate_files()

        self.path_mapping = self.get_path_mapping()

    def _validate_files(self):
        """验证所有npy文件是否存在"""
        missing_files = []
        for filename in self.file_list:
            # validate GT
            gt_path = os.path.join(self.root_dir, 'gt', filename)
            if not (os.path.exists(gt_path) and gt_path.endswith('.npy')):
                missing_files.append(gt_path)

            # validate LQ
            for level in self.NOISE_LEVELS:
                rain_path = os.path.join(self.root_dir, 'rain', level, filename)
                if not (os.path.exists(rain_path) and rain_path.endswith('.npy')):
                    missing_files.append(rain_path)

        if missing_files:
            raise FileNotFoundError(f"{len(missing_files)} files not found !, such as: {missing_files[:3]}")

    def get_path_mapping(self):

        path_mapping = []
        for filename in self.file_list:
            # GT
            gt_path = os.path.join(self.root_dir, 'gt', filename)

            # LQ
            for level in self.NOISE_LEVELS:
                rain_paths = os.path.join(self.root_dir, 'rain', level, filename)
                path_mapping.append({
                    'gt_path': gt_path,
                    'lq_path': rain_paths
                })

        return path_mapping

    def __len__(self):
        return len(self.path_mapping)

    def __getitem__(self, idx):
        sample = self.path_mapping[idx]

        # load data
        gt_path = sample['gt_path']
        lq_path = sample['lq_path']
        gt = np.load(gt_path)
        lq = np.load(lq_path)

        results = {
            'sample': lq,
            'target': gt,
            'index': idx
        }

        return self.pipelines(results) if self.pipelines else results

    def evaluate(self, preds: List[np.ndarray], targets: List[np.ndarray], metric: str, indexes: List[int]) -> float:
        """
        Args:
            preds (List[np.ndarray]): 模型输出列表，每个元素形状为 (B, C, H, W)
            targets (List[np.ndarray]): 真实标签列表，每个元素形状为 (B, C, H, W)
            metric (str): 指标类型，支持 'PSNR' 或 'SSIM'
            indexes (List[int]): index of original data
        Returns:
            float: 所有样本的平均指标值
        """
        assert len(preds) == len(targets), "preds 和 targets 列表长度必须一致"
        metric = metric.upper()
        assert metric in ['PSNR', 'SSIM'], f"不支持的指标类型: {metric}"

        all_scores = []
        logger = get_root_logger()

        for pred_batch, target_batch, index_batch in zip(preds, targets, indexes):
            # 确保 batch 维度一致
            assert pred_batch.shape == target_batch.shape, "pred 和 target 形状必须一致"

            B, C, H, W = pred_batch.shape

            # 遍历 batch 中的每一个样本
            for i in range(B):
                pred = pred_batch[i]  # shape: (C, H, W)
                target = target_batch[i]
                idx = index_batch[i]

                # 转换为 (H, W, C) 格式以适配 skimage 的输入要求
                if C == 1:
                    # 灰度图
                    pred_img = pred[0]
                    target_img = target[0]
                else:
                    # 彩色图
                    pred_img = np.transpose(pred, (1, 2, 0))
                    target_img = np.transpose(target, (1, 2, 0))

                # 计算指标
                if metric == 'PSNR':
                    score = psnr(target_img, pred_img, data_range=target_img.max() - target_img.min())
                elif metric == 'SSIM':
                    multichannel = True
                    if C == 1:
                        multichannel = False

                    score = ssim(target_img, pred_img, multichannel=multichannel,
                                 channel_axis=2 if multichannel else None,
                                 data_range=target_img.max() - target_img.min())
                else:
                    raise ValueError(f"未知指标: {metric}")

                all_scores.append(score)
                # log
                logger.info(f'validating:{self.get_filename(idx)}    '
                            f'metric: {metric}   '
                            f'result: {score}')

        # 返回平均值
        final_score = np.mean(all_scores)
        logger.info(f'mean score: {final_score}')
        return final_score

    def get_filename(self, idx):
        path = self.path_mapping[idx]['gt_path']
        return os.path.basename(path)


@DeprecationWarning
@DATASET.register_module()
class MultiLevelRainHSIDatasetV0(Dataset):
    RAIN_TYPES = ['small', 'medium', 'heavy', 'storm']  # Four noise types are fixed

    def __init__(self,
                 data_root: str,
                 pipelines: Dict = None):
        """
        Args:
            data_root: The root directory of the dataset
        """
        self.root = data_root
        self.pipelines = Compose(pipelines) if pipelines else None

        # Check the directory structure
        self._validate_directory_structure()

        # Load all sample paths
        self.gt_paths = sorted(glob.glob(os.path.join(data_root, 'gt', '*.npy')))
        if not self.gt_paths:
            raise FileNotFoundError(f"No .npy files found in {os.path.join(data_root, 'gt')}")

        self.paths = self._build_paths_index()

    def _validate_directory_structure(self):
        """Verify that the directory structure is as expected"""
        required_dirs = ['gt'] + [os.path.join('rain', t) for t in self.RAIN_TYPES]
        for d in required_dirs:
            if not os.path.isdir(os.path.join(self.root, d)):
                raise RuntimeError(f"Missing required directory: {d}")

    def _build_paths_index(self) -> List[Dict[str, str]]:
        """Build a list of sample index dictionaries"""
        paths = []
        for gt_path in self.gt_paths:
            sample = {'gt_path': gt_path}

            # Obtain the corresponding noise file path
            filename, ext = os.path.basename(gt_path).split('.')
            for rain_type in self.RAIN_TYPES:
                rain_path = os.path.join(self.root, 'rain', rain_type, f'{filename}_{rain_type}.{ext}')
                if not os.path.exists(rain_path):
                    raise FileNotFoundError(f"Missing corresponding rain file: {rain_path}")
                sample['lq_path'] = rain_path
                paths.append(sample.copy())

        return paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.paths[idx]

        # 加载数据
        gt_path = sample['gt_path']
        lq_path = sample['lq_path']
        gt = np.load(gt_path)
        lq = np.load(lq_path)

        results = {
            'sample': lq,
            'target': gt,
            'index': idx
        }

        return self.pipelines(results) if self.pipelines else results

    def get_rain_types(self) -> List[str]:
        """获取支持的噪声类型列表"""
        return self.RAIN_TYPES.copy()
