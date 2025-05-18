from torch.utils.data import Dataset

from .builder import DATASET
from .pipelines import Compose
from .load import read_pts, read_txt, read_npy
from ..utils.evaluate import regression_eval
from ..utils.misc import strings_to_list


@DATASET.register_module()
class SoilDataset2d(Dataset):
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


@DATASET.register_module()
class SoilDataset1d(Dataset):
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
