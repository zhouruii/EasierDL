import torch
import torch.nn as nn

from ..builder import MODEL


# channel refinement block
class CRB(nn.Module):
    def __init__(self, in_channels, r=2):
        super(CRB, self).__init__()
        self.branch_layer = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                          nn.Conv2d(in_channels=in_channels,
                                                    out_channels=in_channels // r,
                                                    kernel_size=1),
                                          nn.ReLU(),
                                          nn.Conv2d(in_channels=in_channels // r,
                                                    out_channels=in_channels,
                                                    kernel_size=1),
                                          nn.Sigmoid())

    def forward(self, x):
        xb = self.branch_layer(x)
        return x * xb


# residual channel refinement block
class RCRB(nn.Module):
    def __init__(self, r=2):
        super(RCRB, self).__init__()
        self.banch_layer = nn.Sequential(nn.Conv2d(in_channels=32,
                                                   out_channels=32,
                                                   kernel_size=1),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=32,
                                                   out_channels=32,
                                                   kernel_size=3,
                                                   padding=1),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=32,
                                                   out_channels=32,
                                                   kernel_size=1),
                                         CRB(in_channels=32,
                                             r=r))

    def forward(self, x):
        xb = self.banch_layer(x)
        return x + xb


# 3 residual channel refinement blocks
class TRCRB(nn.Module):
    def __init__(self, r):
        super(TRCRB, self).__init__()
        self.b1 = RCRB(r=r)
        self.b2 = RCRB(r=r)
        self.b3 = RCRB(r=r)

    def forward(self, x):
        x1 = self.b1(x)
        x2 = self.b2(x1)
        x3 = self.b3(x2)
        return x1, x2, x3


# feature fusion block
class FFB(nn.Module):
    def __init__(self):
        super(FFB, self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(in_channels=96 * 4,
                                             out_channels=96,
                                             kernel_size=1),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=96,
                                             out_channels=32,
                                             kernel_size=3,
                                             padding=1),
                                   nn.ReLU())

    def forward(self, x1, x2, x3, x4):
        x = torch.concat([x1, x2, x3, x4], dim=1)
        x = self.layer(x)
        return x


@MODEL.register_module()
class RSDehazeNet(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel

        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.RCRBs = nn.ModuleList([RCRB() for _ in range(12)])
        self.CRBs = nn.ModuleList([CRB(in_channels=32 * 3) for _ in range(4)])
        self.fusion = FFB()
        self.last_crb = CRB(in_channels=32)
        self.last_conv = nn.Conv2d(32, in_channel, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        res = x
        x = self.conv2(x)
        feats = []
        for i in range(4):
            stage = []
            for j in range(3):
                rcrb_module = self.RCRBs[i + j]
                x = rcrb_module(x)
                stage.append(x)
            stage_feature = torch.concat(stage, dim=1)
            crb_module = self.CRBs[i]
            stage_feature = crb_module(stage_feature)
            feats.append(stage_feature)
        out = self.fusion(*feats)
        out = out + res
        out = self.last_crb(out)
        out = self.last_conv(out)

        return out


if __name__ == '__main__':
    x = torch.randn(4, 305, 256, 256)
    net = RSDehazeNet(in_channel=305)
    y = net(x)
    print(y.shape)
