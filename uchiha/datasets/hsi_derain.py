import glob
import os
from typing import List, Dict

import numpy as np
from torch.utils.data import Dataset

from .builder import DATASET
from .pipelines import Compose


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
