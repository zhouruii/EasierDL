from torch.utils.data import DataLoader

from ..utils import get_root_logger
from ..utils import Registry

DATASET = Registry('dataset')

PIPELINES = Registry('pipelines')

SPECTRAL_DATASET = DATASET


def build_dataset(cfg):
    """ build dataset based on configuration

    Args:
        cfg (dict): Configuration information, where the first key is type

    Returns:
        Dataset: the built optimizer

    """
    logger = get_root_logger()
    logger.info("start building dataset...")

    dataset = SPECTRAL_DATASET.build(cfg)

    logger.info('success!')
    return dataset


def build_dataloader(dataset, cfg):
    """ build dataloader based on configuration

    Args:
        dataset (Dataset): the dataset already built before
        cfg (dict): Configuration information, where the first key is type

    Returns:
        Dataloader: the built dataloader

    """
    logger = get_root_logger()
    logger.info("start building dataloader...")

    dataloader = DataLoader(dataset, **cfg)
    logger.info('success!')

    return dataloader
