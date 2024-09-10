from torch.nn import MSELoss, L1Loss, CrossEntropyLoss

from .builder import CRITERION

CRITERION.register_module(module=MSELoss)
CRITERION.register_module(module=L1Loss)
CRITERION.register_module(module=CrossEntropyLoss)
