import torch
from pytorch_wavelets import DWTForward
from torch import nn

from ..builder import PREPROCESSOR


@PREPROCESSOR.register_module()
class WaveletTransform2d(nn.Module):
    def __init__(self, scales=1,
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
