from torch.utils.data import DataLoader

from ..utils import get_root_logger
from ..utils import Registry

DATASET = Registry('dataset')

SPECTRAL_DATASET = DATASET


def build_dataset(cfg):
    logger = get_root_logger()
    logger.info("start building dataset...")

    dataset = SPECTRAL_DATASET.build(cfg)

    logger.info('success!')
    return dataset


def build_dataloader(dataset, kwargs):
    logger = get_root_logger()
    logger.info("start building dataloader...")

    dataloader = DataLoader(dataset, **kwargs)
    logger.info('success!')

    return dataloader
