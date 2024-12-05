import argparse
from datetime import datetime
from os.path import join

from tensorboardX import SummaryWriter
from torch import nn

from uchiha.apis import train_by_epoch, validate, set_random_seed
from uchiha.datasets.builder import build_dataset, build_dataloader
from uchiha.models.builder import build_model
from uchiha.cores.builder import build_criterion, build_optimizer, build_scheduler

from uchiha.utils import count_parameters, load_config, get_root_logger, print_log, save_checkpoint, \
    load_checkpoint, auto_resume_helper


def parse_args():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--seed', type=int, default=49)
    args_parser.add_argument('--config', '-c', type=str, default='configs/spectral/exp.yaml')
    args_parser.add_argument('--gpu_ids', nargs='+', default=['0'])
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
    logger.info(f'Config:\n{cfg}')

    # random seed
    set_random_seed(args.seed)
    logger.info(f'set random seed= {args.seed}')

    # dataset & dataloader
    trainset = build_dataset(cfg.data.train.dataset.to_dict())
    trainloader = build_dataloader(trainset, cfg.data.train.dataloader.to_dict())

    valset = build_dataset(cfg.data.val.dataset.to_dict())
    valloader = build_dataloader(valset, cfg.data.val.dataloader.to_dict())

    # model
    model = build_model(cfg.model.to_dict())
    if len(args.gpu_ids) > 1:
        device_ids = [int(i) for i in args.gpu_ids]
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
        logger.info(f'Use GPUs: {device_ids}')
    else:
        model = model.cuda()
    total_params = count_parameters(model)
    logger.info(f'total_params: {total_params}')

    # loss function
    criterion = build_criterion(cfg.loss.to_dict())

    # optimizer & scheduler
    optimizer = build_optimizer(model.parameters(), cfg.optimizer.to_dict())
    scheduler = build_scheduler(optimizer, cfg.scheduler.to_dict())

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
        meta = load_checkpoint(resume, model, optimizer)
        start_epoch = meta.get('epoch', 0)
        logger.info(f'checkpoint:{resume} was loaded successfully, start_epoch: {start_epoch + 1}')
    else:
        start_epoch = 0
        logger.info(f'no checkpoint was loaded! start_epoch: {start_epoch + 1}')

    # train & val
    print_freq = cfg.train.print_freq
    val_freq = cfg.val.val_freq
    metric = cfg.val.metric
    save_freq = cfg.checkpoint.save_freq
    total_epoch = cfg.train.epoch
    logger.info('start training...')

    for epoch in range(start_epoch, total_epoch):
        # train
        writer, model, optimizer, scheduler = (
            train_by_epoch(epoch, print_freq, trainloader, model, optimizer, scheduler, criterion, writer))

        # val
        if (epoch + 1) % val_freq == 0:
            print_log(f'epoch:{epoch + 1}/{total_epoch}, validate...', logger)
            _ = validate(epoch, valloader, model, writer, metric)

        # save checkpoint
        if (epoch + 1) % save_freq == 0:
            logger.info(f'saving checkpoint in epoch: {epoch + 1}')
            meta = dict(epoch=epoch + 1)
            save_checkpoint(model, optimizer, join(f'{work_dir}/checkpoints', f'{epoch + 1}.pth'), meta)

    writer.close()


if __name__ == '__main__':
    main()
