from torch.utils.data import Dataset
import torch.nn.functional as F

from .builder import SPECTRAL_DATASET
from .piplines import Compose
from .utils import read_npy, read_txt
from ..utils.misc import strings_to_list


@SPECTRAL_DATASET.register_module()
class SpectralDataset2d(Dataset):

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
        if elements is None:
            return self.ELEMENTS
        if isinstance(elements, str):

            element_names = strings_to_list(elements)

        elif isinstance(elements, (tuple, list)):
            element_names = elements
        else:
            raise ValueError(f'Unsupported type {type(elements)} of elements.')

        return element_names


@SPECTRAL_DATASET.register_module()
class SpectralDataset1d(Dataset):
    ELEMENTS = ['Zn', 'Subs']

    def __init__(self, data_root, gt_path):
        self.data_root = data_root
        self.spectral_data = read_npy(data_root)
        self.gt = read_txt(gt_path)

    def __getitem__(self, index):
        result = dict(
            spectral_data=self.spectral_data[index],
            target=self.gt[index]
        )
        return result

    def __len__(self):
        return len(self.spectral_data)
