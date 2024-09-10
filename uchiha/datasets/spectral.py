from torch.utils.data import Dataset
import torch.nn.functional as F

from .builder import SPECTRAL_DATASET
from .utils import read_pts, read_txt


@SPECTRAL_DATASET.register_module()
class SpectralDataset2d(Dataset):
    ELEMENTS = ['Zn', 'Subs']

    def __init__(self, data_root, gt_path):
        self.data_root = data_root
        self.spectral_data = read_pts(data_root)
        self.gt = read_txt(gt_path)

        # padding
        self.spectral_data = [F.pad(data, (0, 1, 0, 1), mode='constant', value=0) for data in self.spectral_data]

    def __getitem__(self, index):
        result = dict(
            spectral_data=self.spectral_data[index],
            target=self.gt[index]
        )
        return result

    def __len__(self):
        return len(self.spectral_data)


@SPECTRAL_DATASET.register_module()
class SpectralDataset1d(Dataset):
    ELEMENTS = ['Zn', 'Subs']

    def __init__(self, data_root, gt_path):
        self.data_root = data_root
        self.spectral_data = read_pts(data_root, is1d=True)
        self.gt = read_txt(gt_path)

    def __getitem__(self, index):
        result = dict(
            spectral_data=self.spectral_data[index],
            target=self.gt[index]
        )
        return result

    def __len__(self):
        return len(self.spectral_data)
