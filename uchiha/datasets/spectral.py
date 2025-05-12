import glob
import os
from typing import List, Dict

from torch.utils.data import Dataset

from .builder import SPECTRAL_DATASET
from .pipelines import Compose
from ..utils import read_npy, read_txt, read_pts
from ..utils.evaluate import regression_eval
from ..utils.misc import strings_to_list


@SPECTRAL_DATASET.register_module()
class SpectralDataset2d(Dataset):
    """ the dataset for hyperspectral images

    Args:
        data_root (str): the root directory where the dataset is stored
        gt_path (str): path of GT data
        elements (str): Elements to be predicted, a comma-separated string is parsed into a list
        pipelines (Sequence[dict]): Data processing flow: a sequence
            each element is different data processing configuration information
    """

    ELEMENTS = ['Zn', 'Others']

    def __init__(self, data_root, gt_path, elements=None, pipelines=None):

        self.data_root = data_root
        self.spectral_data = read_npy(data_root)
        self.gt = read_txt(gt_path)
        self.ELEMENTS = self.get_elements(elements)

        # postprocess
        self.pipelines = Compose(pipelines)

    def __getitem__(self, index):
        results = dict(
            sample=self.spectral_data[index],
            target=self.gt[index],
            index=index
        )
        return self.pipelines(results)

    def __len__(self):
        return len(self.spectral_data)

    def get_elements(self, elements):
        """ get the elements to be predicted

        Gets a list of containing elements based on the provided string separated by commas

        Args:
            elements (str): Elements to be predicted

        Returns:
            element_names (list):  the list containing elements that need to be predicted

        """
        if elements is None:
            return self.ELEMENTS
        if isinstance(elements, str):

            element_names = strings_to_list(elements)

        elif isinstance(elements, (tuple, list)):
            element_names = elements
        else:
            raise ValueError(f'Unsupported type {type(elements)} of elements.')

        return element_names

    def evaluate(self, preds, targets=None, metric='MAE'):
        """ evaluation of predicted values

        Args:
            preds (Tensor): output prediction of the model
            targets (Tensor): the true value of the predicted element
            metric (str): evaluation metric. Default: 'MAE'

        Returns:
            results (dict): evaluation results for each element

        """
        elements = self.ELEMENTS
        if targets is None:
            targets = self.gt

        results = regression_eval(preds, targets, elements, metric)

        return results


@SPECTRAL_DATASET.register_module()
class SpectralDataset1d(Dataset):
    """ the dataset for hyperspectral sequence-data

    Args:
        data_root (str): the root directory where the dataset is stored
        gt_path (str): path of GT data
        elements (str): Elements to be predicted, a comma-separated string is parsed into a list
        pipelines (Sequence[dict]): Data processing flow: a sequence
            each element is different data processing configuration information
    """

    ELEMENTS = ['Zn', 'Subs']

    def __init__(self, data_root, gt_path, elements=None, pipelines=None):
        self.data_root = data_root
        self.spectral_data = read_pts(data_root, is1d=True)
        self.gt = read_txt(gt_path)
        self.ELEMENTS = self.get_elements(elements)

        self.pipelines = Compose(pipelines) if pipelines else None

    def __getitem__(self, index):
        results = dict(
            sample=self.spectral_data[index],
            target=self.gt[index],
            index=index
        )
        return self.pipelines(results) if self.pipelines else results

    def __len__(self):
        return len(self.spectral_data)

    def get_elements(self, elements):
        """ get the elements to be predicted

        Gets a list of containing elements based on the provided string separated by commas

        Args:
            elements (str): Elements to be predicted

        Returns:
            element_names (list):  the list containing elements that need to be predicted

        """
        if elements is None:
            return self.ELEMENTS
        if isinstance(elements, str):

            element_names = strings_to_list(elements)

        elif isinstance(elements, (tuple, list)):
            element_names = elements
        else:
            raise ValueError(f'Unsupported type {type(elements)} of elements.')

        return element_names

    def evaluate(self, preds, targets=None, metric='MAE'):
        """ evaluation of predicted values

        Args:
            preds (Tensor): output prediction of the model
            targets (Tensor): the true value of the predicted element
            metric (str): evaluation metric. Default: 'MAE'

        Returns:
            results (dict): evaluation results for each element

        """
        elements = self.ELEMENTS
        if targets is None:
            targets = self.gt

        results = regression_eval(preds, targets, elements, metric)

        return results


@SPECTRAL_DATASET.register_module()
class MultiLevelRainHSIDataset(Dataset):
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
        gt = read_npy(gt_path)
        lq = read_npy(lq_path)

        results = {
            'sample': lq,
            'target': gt,
            'index': idx
        }

        return self.pipelines(results) if self.pipelines else results

    def get_rain_types(self) -> List[str]:
        """获取支持的噪声类型列表"""
        return self.RAIN_TYPES.copy()
