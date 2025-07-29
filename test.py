import argparse
from datetime import datetime
from os.path import join

import torch
from torch import nn

from uchiha.apis import set_random_seed, hsi_test
from uchiha.apis.inference_dehaze import remove_module_prefix
from uchiha.datasets.builder import build_dataset, build_dataloader
from uchiha.models.builder import build_model
from uchiha.utils import load_config, get_root_logger


def parse_args():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--seed', type=int, default=49)
    args_parser.add_argument('--config', type=str, default='configs/hsi_dehaze/HD/test/D3.yaml')
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
    checkpoint = torch.load(cfg.checkpoint)
    gpu_ids = [int(i) for i in args.gpu_ids]
    if len(gpu_ids) > 1:
        torch.cuda.set_device(gpu_ids[0])  # 当前上下文绑定主卡
        model = nn.DataParallel(model, device_ids=gpu_ids).cuda(gpu_ids[0])
        device = torch.device(f'cuda:{gpu_ids[0]}')
        logger.info(f'Using GPUs: {gpu_ids}')
        model.load_state_dict(checkpoint['state_dict'])
    else:
        device_id = gpu_ids[0]
        device = torch.device(f'cuda:{device_id}')
        torch.cuda.set_device(device)  # 当前上下文绑定该卡
        model = model.to(device)
        logger.info(f'Using single GPU: {device}')
        model.load_state_dict(remove_module_prefix(checkpoint['state_dict']))
    logger.info(f'checkpoint:{cfg.checkpoint} was loaded successfully!')

    # train & val
    logger.info('start testing...')

    # evaluate
    hsi_test(dataloader=testloader,
             model=model,
             device=device)


if __name__ == '__main__':
    main()
