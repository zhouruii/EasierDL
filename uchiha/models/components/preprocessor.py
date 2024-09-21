import torch
from pytorch_wavelets import DWTForward, DWT1DForward
from torch import nn

from ...utils import sequence_to_image
from ..builder import PREPROCESSOR


@PREPROCESSOR.register_module()
class DWT2d(nn.Module):
    def __init__(self,
                 scales=1,
                 wave='haar',
                 padding='zero'):
        super().__init__()
        self.wavlet_transform = DWTForward(J=scales, wave=wave, mode=padding)

    def forward(self, x):
        # x.shape:B, C, H, W
        out = self.wavlet_transform(x)
        LL, H = out
        LH, HL, HH = H[0][:, :, 0, :, :], H[0][:, :, 1, :, :], H[0][:, :, 2, :, :]
        return torch.cat((LL, LH, HL, HH), dim=1)


@PREPROCESSOR.register_module()
class DWT1d(nn.Module):
    def __init__(self,
                 scales=1,
                 wave='haar',
                 padding='zero',
                 origin=False):
        super().__init__()
        self.wavlet_transform = DWT1DForward(J=scales, wave=wave, mode=padding)
        self.origin = origin

    def forward(self, x):
        # x.shape:B, C, H, W
        if len(x.shape) == 4:
            x = x.flatten(2).transpose(1, 2)

        L, H = self.wavlet_transform(x)

        parallel = []
        parallel.append(sequence_to_image(L))
        for idx, h in enumerate(H):
            parallel.append(sequence_to_image(h))

        if self.origin:
            return x, parallel
        else:
            return parallel


if __name__ == '__main__':
    dwt = DWT1DForward(wave='db6', J=3)
    X = torch.randn(10, 5, 100)
    yl, yh = dwt(X)
