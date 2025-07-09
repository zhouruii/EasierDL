import argparse
from datetime import datetime
from os.path import join

import torch
from tensorboardX import SummaryWriter
from torch import nn

from uchiha.apis import train_by_epoch, validate, set_random_seed, log_model_parameters, unwrap_model
from uchiha.cores.builder import build_criterion, build_optimizer, build_scheduler
from uchiha.datasets.builder import build_dataset, build_dataloader
from uchiha.models.builder import build_model
from uchiha.utils import load_config, get_root_logger, print_log, save_checkpoint, \
    load_checkpoint, auto_resume_helper, log_env_info, get_env_info
from uchiha.utils.logger import ETACalculator


def parse_args():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--seed', type=int, default=49)
    args_parser.add_argument('--config', '-c', type=str, default='configs/hdr_former/v0.yaml')
    args_parser.add_argument('--gpu_ids', nargs='+', default=['0'])
    args_parser.add_argument('--analyze_params', '-ap', type=int, default=0)
    args_parser.add_argument('--multi-process', '-mp', action='store_true')
    args_parser.add_argument('--no_validate', '-n', action='store_true')

    return args_parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # log: tensorboard & logger
    work_dir = cfg.work_dir
    log_time = datetime.now().strftime("%Y-%m-%d-%H-%M")

    writer = SummaryWriter(log_dir=join(f'{work_dir}/tb_loggers', log_time))

    logger = get_root_logger(log_file=join(f'{work_dir}/logs', f'{log_time}.log'))
    log_env_info(logger, get_env_info())
    logger.info(f'Config:\n{cfg}')

    # random seed
    set_random_seed(args.seed)
    logger.info(f'set random seed= {args.seed}')

    # dataset & dataloader
    trainset = build_dataset(cfg.data.train.dataset.to_dict(), phase='train')
    trainloader = build_dataloader(trainset, cfg.data.train.dataloader.to_dict(), phase='train')

    valset = build_dataset(cfg.data.val.dataset.to_dict(), phase='val')
    valloader = build_dataloader(valset, cfg.data.val.dataloader.to_dict(), phase='val')

    # model
    model = build_model(cfg.model.to_dict())
    gpu_ids = [int(i) for i in args.gpu_ids]
    if len(gpu_ids) > 1:
        torch.cuda.set_device(gpu_ids[0])  # 当前上下文绑定主卡
        model = nn.DataParallel(model, device_ids=gpu_ids).cuda(gpu_ids[0])
        device = torch.device(f'cuda:{gpu_ids[0]}')
        print(f'Using GPUs: {gpu_ids}')
    else:
        device_id = gpu_ids[0]
        device = torch.device(f'cuda:{device_id}')
        torch.cuda.set_device(device)  # 当前上下文绑定该卡
        model = model.to(device)
        print(f'Using single GPU: {device}')
    log_model_parameters(unwrap_model(model), logger, max_depth=args.analyze_params)

    # loss function
    criterion = build_criterion(cfg.train.loss.to_dict())

    # optimizer & scheduler
    optimizer = build_optimizer(model.parameters(), cfg.train.optimizer.to_dict())
    scheduler = build_scheduler(optimizer, cfg.train.scheduler.to_dict())

    # resume
    logger.info('start loading checkpoint...')
    auto_resume = cfg.checkpoint.auto_resume
    resume_from = cfg.checkpoint.resume_from
    if auto_resume:
        resume = auto_resume_helper(f'{work_dir}/checkpoints')
    else:
        if resume_from:
            resume = join(f'{work_dir}/checkpoints', f'{resume_from}.pth')
        else:
            resume = None
    if resume:
        meta = load_checkpoint(resume, model, optimizer, scheduler=scheduler)
        start_epoch = meta.get('epoch', 0)
        logger.info(f'checkpoint:{resume} was loaded successfully, start_epoch: {start_epoch + 1}')
    else:
        start_epoch = 0
        logger.info(f'no checkpoint was loaded! start_epoch: {start_epoch + 1}')

    # train & val
    val_freq = cfg.val.val_freq
    metric = cfg.val.metric
    save_freq = cfg.checkpoint.save_freq
    total_epoch = cfg.train.total_epoch
    eta_calc = ETACalculator(total_steps=total_epoch * len(trainloader))
    logger.info('start training...')
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, total_epoch):
        # train
        writer, model, optimizer, scheduler = (
            train_by_epoch(cfg, epoch, trainloader, model, optimizer, scheduler, criterion, writer,
                           eta_calc, device))

        # val
        if (epoch + 1) % val_freq == 0:
            print_log(f'epoch:[{epoch + 1}/{total_epoch}]\tstart validating...', logger)
            _ = validate(epoch, valloader, model, writer, metric, device)

        # save checkpoint
        if (epoch + 1) % save_freq == 0:
            logger.info(f'saving checkpoint in epoch: {epoch + 1}')
            meta = dict(epoch=epoch + 1)
            save_checkpoint(model, optimizer, join(f'{work_dir}/checkpoints', f'{epoch + 1}.pth'), scheduler=scheduler,
                            meta=meta)

    writer.close()


if __name__ == '__main__':
    main()
