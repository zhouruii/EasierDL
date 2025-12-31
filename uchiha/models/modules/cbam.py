# implement of Paper：CBAM: Convolutional Block Attention Module
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

from .common import conv3x3
from ..builder import MODULE, build_module

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class CAM(nn.Module):
    """ Channel Attention Module based on Conv

    Args:
        in_planes (int): number of input channels
        ratio (int): The ratio of the hidden layer to the input dimension when extracting the channel mask
    """

    def __init__(self, in_planes, ratio=16):
        super(CAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SAM(nn.Module):
    """ Spatial Attention Module based on Conv

    Args:
        kernel_size (): kernel size of the convolution operation performed when extracting the spatial mask
    """

    def __init__(self, kernel_size=7):
        super(SAM, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


@MODULE.register_module()
class BasicCAM(nn.Module):
    """ basic Channel Attention Module(CAM)

    Args:
        in_channel (int): number of input channels
        out_channel (int): number of output channels
        stride (int): the stride for the convolution operation. Default: 1
        downsample (dict): operations performed when residual is applied. Default: None
    """

    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicCAM, self).__init__()
        self.conv1 = conv3x3(in_channel, out_channel, stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.ca = CAM(out_channel)

        self.downsample_opt = build_module(downsample)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out

        if self.downsample_opt is not None:
            residual = self.downsample_opt(x)

        out += residual
        out = self.relu(out)

        return out


@MODULE.register_module()
class BasicSAM(nn.Module):
    """ basic Spatial Attention Module(SAM)

    Args:
        in_channel (int): number of input channels
        out_channel (int): number of output channels
        stride (int): the stride for the convolution operation. Default: 1
        downsample (dict): operations performed when residual is applied. Default: None
    """

    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicSAM, self).__init__()
        self.conv1 = conv3x3(in_channel, out_channel, stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.sa = SAM()

        self.downsample_opt = build_module(downsample)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.sa(out) * out

        if self.downsample_opt is not None:
            residual = self.downsample_opt(x)

        out += residual
        out = self.relu(out)

        return out


class BasicCBAM(nn.Module):
    """ Conv*2 ==> CAM ==> SAM ==>

    Args:
        inplanes (int): number of input channels
        planes (int): number of output channels
        stride (int): the stride for the convolution operation. Default: 1
        downsample (nn.Module): operations performed when residual is applied. Default: None
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicCBAM, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = CAM(planes)
        self.sa = SAM()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class CBAMBottleneck(nn.Module):
    """ Conv*3 ==> CAM ==> SAM ==>

    the actual number of output channels is expanded by expansion

    Args:
        inplanes (int): number of input channels
        planes (int): number of output channels
        stride (int): the stride for the convolution operation. Default: 1
        downsample (nn.Module): operations performed when residual is applied. Default: None
    """

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(CBAMBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.ca = CAM(planes * self.expansion)
        self.sa = SAM()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model based on CBAM.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicCBAM, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet18'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet34_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model based on CBAM.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicCBAM, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet34'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet50_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model based on CBAM.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(CBAMBottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet101_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model based on CBAM.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(CBAMBottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet101'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet152_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model based on CBAM.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(CBAMBottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet152'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


class GlobalMinPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        noise = x.view(x.size(0), x.size(1), -1).min(dim=-1, keepdim=True)[0]
        return noise.unsqueeze(-1)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


@MODULE.register_module()
class BCAM(nn.Module):
    def __init__(self, in_channels, ratio=0.5, min_proj=False, act='ReLU', residual=False, min_pool=True):
        super(BCAM, self).__init__()
        self.first_dwconv = conv3x3(in_channels, in_channels, groups=in_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.min_pool = GlobalMinPool2d() if min_pool else None

        self.proj = nn.Sequential(nn.Conv2d(in_channels, int(in_channels * ratio), 1, bias=False),
                                  nn.ReLU(),
                                  nn.Conv2d(int(in_channels * ratio), in_channels, 1, bias=False))
        if min_proj:
            self.proj_for_min = nn.Conv2d(in_channels, in_channels, 1, groups=in_channels)
        else:
            self.proj_for_min = None

        if act == 'ReLU':
            self.act = nn.ReLU()
        elif act == 'Sigmoid':
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(f'activation function:{act} not supported yet !')

        self.residual = residual
        self.post_act = nn.LeakyReLU() if residual else None

    def forward(self, x):
        # x.shape (B, C, H, W)
        residual = x
        x = self.first_dwconv(x)

        avg_out = self.proj(self.avg_pool(x))
        max_out = self.proj(self.max_pool(x))
        out = avg_out + max_out

        if self.min_pool:
            if self.proj_for_min:
                min_out = self.proj_for_min(self.min_pool(x))
            else:
                min_out = self.min_pool(x)
        else:
            min_out = 0

        out -= min_out
        att = self.act(out)

        if self.residual:
            return self.post_act(residual + att * x)
        else:
            return att * x


class BasicECA1d(nn.Module):
    """Constructs an ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(BasicECA1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.channel = channel
        self.k_size = k_size

    def forward(self, x):
        # b hw c
        # feature descriptor on the global spatial information
        y = self.avg_pool(x.transpose(-1, -2))

        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2))

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

    def flops(self):
        flops = 0
        flops += self.channel * self.channel * self.k_size

        return flops
