import os
from os import symlink, remove
from os.path import dirname, join, exists, basename

import torch
from torch.optim import Optimizer


def save_checkpoint(model: torch.nn.Module,
                    optimizer: Optimizer,
                    filepath: str,
                    meta=None) -> None:
    if meta is None:
        meta = {}

    checkpoint = {
        'meta': meta,
        'state_dict': model.state_dict(),
    }

    # save optimizer state dict in the checkpoint
    if isinstance(optimizer, Optimizer):
        checkpoint['optimizer'] = optimizer.state_dict()
    elif isinstance(optimizer, dict):
        checkpoint['optimizer'] = {}
        for name, optim in optimizer.items():
            checkpoint['optimizer'][name] = optim.state_dict()

    # save
    torch.save(checkpoint, filepath)


def load_checkpoint(filename: str,
                    model: torch.nn.Module,
                    optimizer: Optimizer, ) -> dict:
    # load
    checkpoint = torch.load(filename, map_location='cpu')

    # meta
    meta = checkpoint['meta']

    # uchiha & load params
    model.load_state_dict(checkpoint['state_dict'])

    # optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])

    # clean memory
    del checkpoint
    torch.cuda.empty_cache()

    return meta


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file