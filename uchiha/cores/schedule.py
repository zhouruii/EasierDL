from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from .builder import SCHEDULER

SCHEDULER.register_module(module=StepLR)
SCHEDULER.register_module(module=CosineAnnealingLR)
