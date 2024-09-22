import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss, L1Loss, CrossEntropyLoss, HuberLoss

from .builder import CRITERION

CRITERION.register_module(module=MSELoss)
CRITERION.register_module(module=L1Loss)
CRITERION.register_module(module=CrossEntropyLoss)
CRITERION.register_module(module=HuberLoss)


@CRITERION.register_module()
class MultiL1Loss(nn.Module):
    def __init__(self,
                 weights=2):
        """ multi-branch L1Loss

        When the network has multiple pipelines, the loss is calculated for
        each pipeline and then these losses are weighted and constitute the final loss.

        Args:
            weights (List[int] | int): weights of each pipeline
                When a list is provided, the weighted sum is based on the elements in the list,
                when an integer is provided, the weights are randomly initialized and updated with backpropagation.
                Default: 2
        """
        super(MultiL1Loss, self).__init__()
        self.weights = weights

    def forward(self, prediction, target):
        loss = 0
        for idx, pred in enumerate(prediction):
            loss += F.l1_loss(pred, target) * self.weights[idx]

        return loss

