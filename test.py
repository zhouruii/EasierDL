import argparse
from datetime import datetime
from os.path import join

import torch
from torch import nn

from uchiha.apis import set_random_seed, log_model_parameters, unwrap_model, simple_test
from uchiha.datasets.builder import build_dataset, build_dataloader
from uchiha.models.builder import build_model
from uchiha.utils import load_config, get_root_logger


def parse_args():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--seed', type=int, default=49)
    args_parser.add_argument('--config', type=str, default='configs/hdr_former/AVIRIS/test.yaml')
    args_parser.add_argument('--checkpoint', '-c', default='experiments/hdr_former/AVIRIS/baseline/checkpoints/100.pth')
    args_parser.add_argument('--gpu_ids', nargs='+', default=['0'])

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
    testset = build_dataset(cfg.data.test.dataset.to_dict(), phase='test')
    testloader = build_dataloader(testset, cfg.data.test.dataloader.to_dict(), phase='test')

    # model
    model = build_model(cfg.model.to_dict())
    if len(args.gpu_ids) > 1:
        device_ids = [int(i) for i in args.gpu_ids]
        torch.cuda.set_device(device_ids[0])
        model = nn.DataParallel(model.cuda(), device_ids=device_ids)
        logger.info(f'Use GPUs: {device_ids}')
    else:
        model = model.cuda()
    log_model_parameters(unwrap_model(model), logger, max_depth=1)

    # resume
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    logger.info(f'checkpoint:{args.checkpoint} was loaded successfully!')

    # train & val
    metric = cfg.metric
    logger.info('start testing...')

    # evaluate
    results = simple_test(dataloader=testloader,
                          model=model,
                          metric=metric)

    print(results)


if __name__ == '__main__':
    main()
