import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss, L1Loss, CrossEntropyLoss

from .builder import CRITERION

CRITERION.register_module(module=MSELoss)
CRITERION.register_module(module=L1Loss)
CRITERION.register_module(module=CrossEntropyLoss)


@CRITERION.register_module()
class MultiL1Loss(nn.Module):
    def __init__(self,weights):
        super(MultiL1Loss, self).__init__()
        self.weights = weights

    def forward(self, prediction, target):
        # Example: Mean squared error with an additional penalty
        loss = 0
        for idx, pred in enumerate(prediction):
            loss += F.l1_loss(pred, target) * self.weights[idx]

        return loss
