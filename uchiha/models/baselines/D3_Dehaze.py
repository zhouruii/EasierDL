"""
D3-Dehaze: a divide-and-conquer framework for enhanced single image dehazing
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init

from ..builder import MODEL


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class PSA(nn.Module):
    def __init__(self, channel):
        super().__init__()

        reduction = 4
        S = 4
        self.S = S

        self.convs = nn.ModuleList([
            nn.Conv2d(channel // S, channel // S, kernel_size=2 * (i + 1) + 1, padding=i + 1)
            for i in range(S)
        ])

        self.se_blocks = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channel // S, channel // (S * reduction), kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // (S * reduction), channel // S, kernel_size=1, bias=False),
                nn.Sigmoid()
            ) for i in range(S)
        ])

        self.softmax = nn.Softmax(dim=1)

        self.final_conv = nn.Conv2d(channel, channel, kernel_size=1, bias=False)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.size()

        SPC_out = x.view(b, self.S, c // self.S, h, w)
        # for idx, conv in enumerate(self.convs):
        #     SPC_out[:, idx, :, :, :] = conv(SPC_out[:, idx, :, :, :])

        # 用列表存储每组卷积结果
        outputs = []
        for idx, conv in enumerate(self.convs):
            part = SPC_out[:, idx, :, :, :]  # [b, c//S, h, w]
            out = conv(part)  # 卷积后形状一致
            outputs.append(out.unsqueeze(1))  # [b, 1, c//S, h, w]

        # 拼回 [b, S, c//S, h, w]
        SPC_out_new = torch.cat(outputs, dim=1)

        se_out = []
        for idx, se in enumerate(self.se_blocks):
            se_out.append(se(SPC_out[:, idx, :, :, :]))
        SE_out = torch.stack(se_out, dim=1)
        SE_out = SE_out.expand_as(SPC_out)

        softmax_out = self.softmax(SE_out)

        PSA_out = SPC_out * softmax_out
        PSA_out = PSA_out.view(b, -1, h, w)

        out = self.final_conv(PSA_out)
        return out


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class MCP(nn.Module):
    def __init__(self, conv, dim, kernel_size, ):
        super(MCP, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.PSA = PSA(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.PSA(res)
        res = self.palayer(res)
        res += x
        return res


class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, mcps):
        super(Group, self).__init__()
        modules = [MCP(conv, dim, kernel_size) for _ in range(mcps)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gp(x)
        res += x
        return res


class PR(nn.Module):
    def __init__(self, dim, conv):
        super(PR, self).__init__()
        f = dim // 4
        self.conv1 = conv(dim, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m


class FSA(nn.Module):
    def __init__(self, num_fea):
        super(FSA, self).__init__()
        self.channel1 = num_fea // 2
        self.channel2 = num_fea - self.channel1
        self.convblock = nn.Sequential(
            nn.Conv2d(self.channel1, self.channel1, 3, 1, 1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(self.channel1, self.channel1, 3, 1, 1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(self.channel1, self.channel1, 3, 1, 1),
        )
        self.A_att_conv = CALayer(self.channel1)
        self.B_att_conv = CALayer(self.channel2)

        self.fuse1 = nn.Conv2d(num_fea, self.channel1, 1, 1, 0)
        self.fuse2 = nn.Conv2d(num_fea, self.channel2, 1, 1, 0)
        self.fuse = nn.Conv2d(num_fea, num_fea, 1, 1, 0)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.channel1, self.channel2], dim=1)
        x1 = self.convblock(x1)
        A = self.A_att_conv(x1)
        P = torch.cat((x2, A * x1), dim=1)
        B = self.B_att_conv(x2)
        Q = torch.cat((x1, B * x2), dim=1)
        c = torch.cat((self.fuse1(P), self.fuse2(Q)), dim=1)
        out = self.fuse(c)
        return out


class AF(nn.Module):
    def __init__(self, num_fea):
        super(AF, self).__init__()
        self.CA1 = CALayer(num_fea)
        self.CA2 = CALayer(num_fea)
        self.fuse = nn.Conv2d(num_fea * 2, num_fea, 1)

    def forward(self, x1, x2):
        x1 = self.CA1(x1) * x1
        x2 = self.CA2(x2) * x2
        return self.fuse(torch.cat((x1, x2), dim=1))


class HD(nn.Module):
    def __init__(self, num_fea):
        super(HD, self).__init__()
        self.CB1 = FSA(num_fea)
        self.CB2 = FSA(num_fea)
        self.CB3 = FSA(num_fea)
        self.AF1 = AF(num_fea)
        self.AF2 = AF(num_fea)

    def forward(self, x):
        x1 = self.CB1(x)
        x2 = self.CB2(x1)
        x3 = self.CB3(x2)
        f1 = self.AF1(x3, x2)
        f2 = self.AF2(f1, x1)
        return x + f2


@MODEL.register_module()
class D3(nn.Module):
    def __init__(self, in_channels=3, gps=3, mcps=19, conv=default_conv):
        super(D3, self).__init__()
        self.gps = gps
        self.dim = 64
        kernel_size = 3
        pre_process = [conv(in_channels, self.dim, kernel_size)]
        assert self.gps == 3
        self.g1 = Group(conv, self.dim, kernel_size, mcps=mcps)
        self.g2 = Group(conv, self.dim, kernel_size, mcps=mcps)
        self.g3 = Group(conv, self.dim, kernel_size, mcps=mcps)
        self.ca = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * self.gps, self.dim // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // 16, self.dim * self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
        ])
        self.palayer = PALayer(self.dim)
        self.pr = PR(self.dim, nn.Conv2d)
        self.conv_pr = conv(self.dim, self.dim, kernel_size)
        self.HD = HD(self.dim)
        post_precess = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, in_channels, kernel_size)
        ]
        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)

    def forward(self, x1):
        x = self.pre(x1)
        HD = self.HD(x)
        res1 = self.g1(x)
        res2 = self.g2(res1)
        res3 = self.g3(res2)
        w = self.ca(torch.cat([res1, res2, res3], dim=1))
        w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]
        out = w[:, 0, :] * res1 + w[:, 1, :] * res2 + w[:, 2, :] * res3
        out = self.palayer(out) + HD
        out = self.conv_pr(out)
        out = self.pr(out)
        x = self.post(out)
        return x + x1
