from ..utils import Registry, get_root_logger

OPTIMIZER = Registry('optimizer')

CRITERION = Registry('criterion')

SCHEDULER = Registry('scheduler')

logger = get_root_logger()


def build_optimizer(params, cfg):
    logger.info("start building criterion...")

    optimizer = OPTIMIZER.get(cfg.pop('type'))
    optimizer = optimizer(params, **cfg)

    logger.info('success!')
    return optimizer


def build_criterion(cfg):
    logger.info("start building criterion...")

    criterion = CRITERION.build(cfg)

    logger.info('success!')
    return criterion


def build_scheduler(optimizer, cfg):
    logger.info("start building criterion...")

    scheduler = SCHEDULER.get(cfg.pop('type'))
    scheduler = scheduler(optimizer, **cfg)

    logger.info('success!')
    return scheduler
