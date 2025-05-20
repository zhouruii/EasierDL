import random

import numpy as np
import torch

from ..utils import print_log, get_root_logger


def train_by_epoch(epoch, total_epoch, print_freq, dataloader, model, optimizer, scheduler, criterion, writer,
                   eta_calculator):
    """ train for one epoch

    Prints logs based on the configured frequency (based on the number of iterations)

    Args:
        epoch (int): the number of epoch trained
        total_epoch (int): the number of total epoch
        print_freq (int): Frequency of printing logs
        dataloader (torch.utils.data.Dataloader): training set's dataloader
        model (torch.nn.Module): model built from configuration file
        optimizer (class): optimizer built from configuration file
        scheduler (class): lr scheduler built from configuration file
        criterion (class): loss function built from configuration file
        writer (SummaryWriter): tensorboard-based loggers currently support tensorboardX
        eta_calculator (class): ETA (Estimated Time) Calculator

    Returns:
        writer (dict): The updated logger, also return the updated model, optimizer and scheduler.

    """
    model.train()
    for idx, data in enumerate(dataloader):
        # data
        sample = data['sample'].cuda()
        target = data['target'].cuda().float()

        # forward & loss
        pred = model(sample)
        loss = criterion(pred, target)

        # backward & optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        eta = eta_calculator.update()

        # log
        if (idx + 1) % print_freq == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print_log(
                f'epoch:[{epoch + 1}/{total_epoch}]\titer:[{idx + 1}/{len(dataloader)}]\tloss: {loss}\t'
                f'lr:{current_lr}\teta:{eta_calculator.format_eta(eta)}',
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
