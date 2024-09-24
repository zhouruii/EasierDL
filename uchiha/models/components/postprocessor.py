from typing import List

import torch
from pytorch_wavelets import DWT1DInverse, DWTInverse
from torch import nn, Tensor

from ..builder import POSTPROCESSOR


@POSTPROCESSOR.register_module()
class IDWT2d(nn.Module):
    """ 2D Inverse Discrete Wavelet Transform

    Args:
        wave (str): The type of wavelet
    """

    def __init__(self,
                 wave='haar'):
        super().__init__()
        self.IDWT = DWTInverse(wave=wave)

    def forward(self, x):
        # x.shape:B, C, H, W
        out = self.wavlet_transform(x)
        LL, H = out
        LH, HL, HH = H[0][:, :, 0, :, :], H[0][:, :, 1, :, :], H[0][:, :, 2, :, :]
        return torch.cat((LL, LH, HL, HH), dim=1)


@POSTPROCESSOR.register_module()
class IDWT1d(nn.Module):
    """ 1D Inverse Discrete Wavelet Transform

    Args:
        wave (str): The type of wavelet
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
    """

    def __init__(self,
                 wave='haar',
                 in_channel=1024,
                 out_channel=1):
        super().__init__()
        self.IDWT = DWT1DInverse(wave=wave)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_channel, out_channel)

    def forward(self, x):
        L, H = x
        H = (H,)
        out = self.IDWT((L, H))
        pooling = self.pooling(out.transpose(1, 2)).squeeze(2)
        return self.fc(pooling)


@POSTPROCESSOR.register_module()
class WeightedSum(nn.Module):
    """ assign weights to multiple inputs and accumulate

    If a list is provided, it will be weighted according to the value of the list,
    if not, the mean weight will be assigned initially and updated with backward propagation.

    Args:
        weights (): weights of each input
    """

    def __init__(self, weights):

        super().__init__()
        if isinstance(weights, list):
            self.weights = weights
        else:
            self.weights = nn.ParameterList([nn.Parameter(torch.tensor(1 / weights)) for _ in range(weights)])

    def forward(self, x: List[Tensor]):

        result = torch.zeros_like(x[0])
        for idx, parallel in enumerate(x):
            result += self.weights[idx] * parallel

        return result


if __name__ == '__main__':
    dwt = DWT1DInverse(wave='haar')
    X = torch.randn(10, 5, 100)
    yl, yh = dwt(X)
