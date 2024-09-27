import argparse
from datetime import datetime
from os.path import join

import torch
from tensorboardX import SummaryWriter

from uchiha.apis import train_by_epoch, validate, set_random_seed
from uchiha.datasets.builder import build_dataset, build_dataloader
from uchiha.models.builder import build_model
from uchiha.cores.builder import build_criterion, build_optimizer, build_scheduler

from uchiha.utils import count_parameters, load_config, get_root_logger, print_log, save_checkpoint, \
    load_checkpoint, auto_resume_helper


def parse_args():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--seed', type=int, default=49)
    args_parser.add_argument('--config', type=str, default='configs/spectral/Zn/dwt_channel.yaml')
    args_parser.add_argument('--checkpoint', '-c', default='experiment/Zn/dwt_channel/checkpoints/100.pth')

    return args_parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # log: tensorboard & logger
    work_dir = cfg.work_dir
    log_time = datetime.now().strftime("%Y-%m-%d-%H-%M")

    logger = get_root_logger(log_file=join(f'{work_dir}/logs', f'{log_time}.log'))
    logger.info(f'Config:\n{cfg}')

    # random seed
    set_random_seed(args.seed)
    logger.info(f'set random seed= {args.seed}')

    # dataset & dataloader
    testset = build_dataset(cfg.data.val.dataset.to_dict())
    testloader = build_dataloader(testset, cfg.data.val.dataloader.to_dict())

    # model
    model = build_model(cfg.model.to_dict()).cuda()
    total_params = count_parameters(model)
    logger.info(f'total_params: {total_params}')

    # resume
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    logger.info(f'checkpoint:{args.checkpoint} was loaded successfully!')

    # train & val
    metric = cfg.val.metric
    logger.info('start testing...')

    targets = []
    preds = []
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(testloader):
            # data
            sample = data['sample'].cuda()
            targets.append(data['target'])

            # forward
            with torch.no_grad():
                pred = model(sample)
                preds.append(pred)

    # evaluate
    results = testset.evaluate(preds, targets, metric)

    print(results)


if __name__ == '__main__':
    main()
