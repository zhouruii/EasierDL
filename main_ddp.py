# main_ddp.py
import argparse
import os
from datetime import datetime
from os.path import join

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

# from your project
from uchiha.apis import train_by_epoch_ddp, validate_ddp, set_random_seed, log_model_parameters, unwrap_model
from uchiha.cores.builder import build_criterion, build_optimizer, build_scheduler
from uchiha.datasets.builder import build_dataset, build_dataloader
from uchiha.models.builder import build_model
from uchiha.utils import load_config, get_root_logger, print_log, save_checkpoint, \
    load_checkpoint, auto_resume_helper, log_env_info, get_env_info
from uchiha.utils.logger import ETACalculator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=49)
    parser.add_argument('--config', '-c', type=str, default='configs/hdr_former/v0.yaml')
    parser.add_argument('--analyze_params', '-ap', type=int, default=0)
    parser.add_argument('--no_validate', '-n', action='store_true')
    # torchrun sets LOCAL_RANK env, we still accept arg for manual run
    parser.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', 0)))
    return parser.parse_args()


def setup_ddp():
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://')


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    local_rank = args.local_rank
    # world_size from env set by torchrun
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))

    # init ddp
    setup_ddp()
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    # only rank 0 create writers / logger
    if rank == 0:
        work_dir = cfg.work_dir
        log_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
        writer = SummaryWriter(log_dir=join(f'{work_dir}/tb_loggers', log_time))
        logger = get_root_logger(log_file=join(f'{work_dir}/logs', f'{log_time}.log'))
        log_env_info(logger, get_env_info())
        logger.info(f'Config:\n{cfg}')
        logger.info(f'Using DDP world_size={world_size}, rank={rank}, local_rank={local_rank}')
    else:
        writer = None
        logger = None
        work_dir = cfg.work_dir

    # set different seed per process to avoid same augmentation / sampling
    set_random_seed(args.seed + local_rank, deterministic=True)

    # build dataset
    trainset = build_dataset(cfg.data.train.dataset.to_dict(), phase='train')
    valset = build_dataset(cfg.data.val.dataset.to_dict(), phase='val')

    # distributed samplers
    train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(valset, num_replicas=world_size, rank=rank, shuffle=False)

    # dataloader configs (ensure sampler provided; do not use shuffle=True)
    train_dataloader_cfg = cfg.data.train.dataloader.to_dict()
    train_dataloader_cfg.update({
        'sampler': train_sampler,
        'shuffle': False,
        'num_workers': train_dataloader_cfg.get('num_workers', 4),
        'pin_memory': train_dataloader_cfg.get('pin_memory', True),
        'persistent_workers': train_dataloader_cfg.get('persistent_workers', True),
        'drop_last': True
    })

    val_dataloader_cfg = cfg.data.val.dataloader.to_dict()
    val_dataloader_cfg.update({
        'sampler': val_sampler,
        'shuffle': False,
        'num_workers': val_dataloader_cfg.get('num_workers', 2),
        'pin_memory': val_dataloader_cfg.get('pin_memory', True),
        'drop_last': False
    })

    trainloader = build_dataloader(trainset, train_dataloader_cfg, phase='train')
    valloader = build_dataloader(valset, val_dataloader_cfg, phase='val')

    if rank == 0 and logger is not None:
        logger.info(f'dataset: {trainset.__class__.__name__} loaded! items: {len(trainset)}')
        logger.info(f'val dataset: {valset.__class__.__name__} loaded! items: {len(valset)}')

    # build model and move to local device
    model = build_model(cfg.model.to_dict()).to(device)

    # wrap with DDP: each process wraps its local model instance
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    if rank == 0 and logger is not None:
        log_model_parameters(unwrap_model(model), logger, max_depth=args.analyze_params)

    # loss / optimizer / scheduler (optimizer on model.parameters() after wrapping is ok)
    criterion = build_criterion(cfg.train.loss.to_dict()).to(device)
    optimizer = build_optimizer(model.parameters(), cfg.train.optimizer.to_dict())
    scheduler = build_scheduler(optimizer, cfg.train.scheduler.to_dict())

    # resume only on rank 0, then broadcast weights
    start_epoch = 0
    resume = None
    auto_resume = cfg.checkpoint.auto_resume
    resume_from = cfg.checkpoint.resume_from
    if auto_resume and rank == 0:
        resume = auto_resume_helper(f'{work_dir}/checkpoints')
    elif resume_from and rank == 0:
        resume = join(f'{work_dir}/checkpoints', f'{resume_from}.pth')

    if resume and rank == 0:
        meta = load_checkpoint(resume, model, optimizer, scheduler=scheduler)
        start_epoch = meta.get('epoch', 0)
        logger.info(f'checkpoint:{resume} loaded, start_epoch {start_epoch + 1}')

    # make sure all processes wait and (optionally) broadcast model state
    dist.barrier()
    # If only rank0 loaded checkpoint, broadcast params to others
    if rank == 0 and resume:
        # model is already updated on rank0, broadcast will sync others
        pass
    else:
        # other ranks will receive via barrier; to be safe you can explicit broadcast
        for param in model.parameters():
            dist.broadcast(param.data, src=0)

    # training loop
    total_epoch = cfg.train.total_epoch
    val_freq = cfg.val.val_freq
    save_freq = cfg.checkpoint.save_freq
    metric = cfg.val.metric

    if rank == 0:
        eta_calc = ETACalculator(total_steps=total_epoch * len(trainloader))
    else:
        eta_calc = None

    for epoch in range(start_epoch, total_epoch):
        # important: tell sampler the epoch for shuffling
        train_sampler.set_epoch(epoch)

        writer, model, optimizer, scheduler = train_by_epoch_ddp(
            cfg, epoch, trainloader, model, optimizer, scheduler, criterion, writer,
            eta_calc, device, local_rank=local_rank, rank=rank
        )

        # validation only on rank0
        if (epoch + 1) % val_freq == 0 and not args.no_validate:
            if rank == 0:
                print_log(f'epoch:[{epoch + 1}/{total_epoch}]\tstart validating...', logger)
            _ = validate_ddp(epoch, valloader, model, writer, metric, device, rank, world_size)

        if (epoch + 1) % save_freq == 0 and rank == 0:
            logger.info(f'saving checkpoint in epoch: {epoch + 1}')
            meta = dict(epoch=epoch + 1)
            save_checkpoint(model, optimizer, join(f'{work_dir}/checkpoints', f'{epoch + 1}.pth'),
                            scheduler=scheduler, meta=meta)

    if rank == 0 and writer is not None:
        writer.close()

    cleanup_ddp()


if __name__ == '__main__':
    main()
