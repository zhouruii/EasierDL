from torch import nn

from ..utils import get_root_logger
from ..utils import Registry

MODEL = Registry('model')

MODULE = Registry('module')

MODULE.register_module(module=nn.Identity, name='Identity')


def build_module(cfg) -> nn.Module:
    """ build basemodule based on configuration

    Args:
        cfg (dict): Configuration information, where the first key is type

    Returns:
        returns the built basemodule

    """
    if cfg is not None:
        return MODULE.build(cfg)
    else:
        return nn.Identity()


def build_model(cfg) -> nn.Module:
    """ build model based on configuration

    Args:
        cfg (dict): Configuration information, where the first key is type

    Returns:
        returns the built model

    """
    logger = get_root_logger()
    logger.info("start building model...")

    model = MODEL.build(cfg)

    logger.info('success!')
    return model
