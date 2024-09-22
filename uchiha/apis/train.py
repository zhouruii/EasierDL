import random

import numpy as np
import torch

from ..utils import print_log, get_root_logger


def train_by_epoch(epoch, dataloader, model, optimizer, scheduler, criterion, writer):
    """ 训练一轮

    Prints logs based on the configured frequency (based on the number of iterations)

    Args:
        epoch (int): 训练的轮数
        dataloader (torch.utils.data.Dataloader): 构建好的训练数据加载器
        model (torch.nn.Module): 构建好的模型
        optimizer (class): 构建好的优化器
        scheduler (class): 构建好的学习率调度器
        criterion (class): 构建好的损失函数
        writer (SummaryWriter): 基于tensorboard的记录器 目前支持tensorboardX

    Returns:
        writer (dict): 更新记录器 同时会返回更新的模型，优化器与调度器

    """
    model.train()
    for idx, data in enumerate(dataloader):
        # data
        spectral_data = data['sample'].cuda()
        target = data['target'].cuda().float()

        # forward & loss
        pred = model(spectral_data)
        loss = criterion(pred, target)

        # backward & optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log
        # TODO 这里打印日志的频次需要根据配置的值动态调整
        if (idx + 1) % 5 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print_log(f'epoch:[{epoch + 1}], iter:[{idx + 1}/{len(dataloader)}], loss: {loss}, lr:{current_lr}',
                      get_root_logger())

        writer.add_scalar('loss', loss.item(), epoch * len(dataloader) + idx)

    scheduler.step()

    return writer, model, optimizer, scheduler


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
