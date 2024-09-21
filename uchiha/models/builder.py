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


def build_preprocessor(cfg):
    if cfg is not None:
        return PREPROCESSOR.build(cfg)


def build_postprocessor(cfg):
    if cfg is not None:
        return POSTPROCESSOR.build(cfg)


def build_embedding(cfg):
    if cfg is not None:
        return EMBEDDING.build(cfg)


def build_basemodule(cfg):
    if cfg is not None:
        return BASEMODULE.build(cfg)


def build_downsample(cfg):
    if cfg is not None:
        return DOWNSAMPLE.build(cfg)


def build_upsample(cfg):
    if cfg is not None:
        return UPSAMPLE.build(cfg)


def build_bottleneck(cfg):
    if cfg is not None:
        return BOTTLENECK.build(cfg)


def build_head(cfg):
    if cfg is not None:
        return HEAD.build(cfg)


def build_fusion(cfg):
    if cfg is not None:
        return FUSION.build(cfg)


def build_model(cfg):
    logger = get_root_logger()
    logger.info("start building model...")

    model = MODEL.build(cfg)

    logger.info('success!')
    return model
