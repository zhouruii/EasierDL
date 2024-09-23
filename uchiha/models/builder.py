from torch import nn

from ..utils import get_root_logger
from ..utils import Registry

MODEL = Registry('model')

BASEMODULE = Registry('basemodule')
BOTTLENECK = Registry('bottleneck')
DOWNSAMPLE = Registry('downsample')
EMBEDDING = Registry('embedding')
FUSION = Registry('fusion')
HEAD = Registry('head')
PREPROCESSOR = Registry('preprocessor')
POSTPROCESSOR = Registry('postprocessor')
UPSAMPLE = Registry('upsample')


def build_preprocessor(cfg) -> nn.Module:
    """ build preprocessor based on configuration

    Args:
        cfg (dict): Configuration information, where the first key is type

    Returns:
        returns the built preprocessor

    """
    if cfg is not None:
        return PREPROCESSOR.build(cfg)


def build_postprocessor(cfg) -> nn.Module:
    """ build postprocessor based on configuration

    Args:
        cfg (dict): Configuration information, where the first key is type

    Returns:
        returns the built postprocessor

    """
    if cfg is not None:
        return POSTPROCESSOR.build(cfg)


def build_embedding(cfg) -> nn.Module:
    """ build embedding based on configuration

    Args:
        cfg (dict): Configuration information, where the first key is type

    Returns:
        returns the built embedding

    """
    if cfg is not None:
        return EMBEDDING.build(cfg)


def build_basemodule(cfg) -> nn.Module:
    """ build basemodule based on configuration

    Args:
        cfg (dict): Configuration information, where the first key is type

    Returns:
        returns the built basemodule

    """
    if cfg is not None:
        return BASEMODULE.build(cfg)


def build_downsample(cfg) -> nn.Module:
    """ build downsample based on configuration

    Args:
        cfg (dict): Configuration information, where the first key is type

    Returns:
        returns the built downsample

    """
    if cfg is not None:
        return DOWNSAMPLE.build(cfg)


def build_upsample(cfg) -> nn.Module:
    """ build upsample based on configuration

    Args:
        cfg (dict): Configuration information, where the first key is type

    Returns:
        returns the built upsample

    """
    if cfg is not None:
        return UPSAMPLE.build(cfg)


def build_bottleneck(cfg) -> nn.Module:
    """ build bottleneck based on configuration

    Args:
        cfg (dict): Configuration information, where the first key is type

    Returns:
        returns the built bottleneck

    """
    if cfg is not None:
        return BOTTLENECK.build(cfg)


def build_head(cfg) -> nn.Module:
    """ build head based on configuration

    Args:
        cfg (dict): Configuration information, where the first key is type

    Returns:
        returns the built head

    """
    if cfg is not None:
        return HEAD.build(cfg)


def build_fusion(cfg) -> nn.Module:
    """ build fusion based on configuration

    Args:
        cfg (dict): Configuration information, where the first key is type

    Returns:
        returns the built fusion

    """
    if cfg is not None:
        return FUSION.build(cfg)


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
