import math
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, _LRScheduler

from .builder import SCHEDULER

SCHEDULER.register_module(module=StepLR)
SCHEDULER.register_module(module=CosineAnnealingLR)


@SCHEDULER.register_module()
class LinearWarmupCosineLR(_LRScheduler):
    """
    线性预热 + 余弦退火学习率调度器（单类整合版）

    参数:
        optimizer: 优化器对象
        total_epochs: 总训练轮数
        warmup_epochs: 预热轮数（默认总轮数的5%）
        warmup_start_lr: 起始学习率（默认初始lr的1%）
        min_lr: 最小学习率（默认初始lr的1%）
        last_epoch: 恢复训练时使用的参数
    """

    def __init__(self,
                 optimizer,
                 total_epochs,
                 warmup_epochs=None,
                 warmup_start_lr=None,
                 min_lr=None,
                 last_epoch=-1):

        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs if warmup_epochs is not None else max(1, int(total_epochs * 0.05))
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr is not None else optimizer.defaults['lr'] * 0.01
        self.min_lr = min_lr if min_lr is not None else optimizer.defaults['lr'] * 0.01

        # 线性预热的增量
        self.warmup_delta = (optimizer.defaults['lr'] - self.warmup_start_lr) / self.warmup_epochs

        # 余弦退火的参数
        self.cosine_epochs = max(1, total_epochs - self.warmup_epochs)

        super(LinearWarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 线性预热阶段
            curr_lr = self.warmup_start_lr + self.last_epoch * self.warmup_delta
            return [curr_lr for _ in self.base_lrs]

        else:
            # 余弦退火阶段
            cosine_epoch = self.last_epoch - self.warmup_epochs
            progress = cosine_epoch / self.cosine_epochs
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            curr_lr = self.min_lr + (self.base_lrs[0] - self.min_lr) * cosine_decay
            return [curr_lr for _ in self.base_lrs]

    def get_last_lr(self):
        """ 当前学习率（兼容最新版PyTorch） """
        return self.get_lr()
