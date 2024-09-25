__all__ = ['BasicResidualBlock', 'ResidualBottleneck']

from typing import Optional, Callable

from torch import nn, Tensor

from ..builder import BASEMODULE
from ...utils.model import conv3x3, conv1x1


# TODO 待整合
@BASEMODULE.register_module()
class BasicResidualBlock(nn.Module):
    """ basic residual block (RB)

    Args:
        inplanes (int): number of input channels
        planes (int): number of output channels
        stride (int): the stride for the convolution operation. Default: 1
        downsample (nn.Module): operations performed when residual is applied. Default: None
        groups (int): the groups for the convolution operation. Default: 1
        base_width (int): base width of image. Default: 64
        dilation (int): the dilation for the convolution operation. Default: 1
        norm_layer (nn.Module): normalization layer after convolution layer. Default: None
    """

    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:

        super(BasicResidualBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


@BASEMODULE.register_module()
class ResidualBottleneck(nn.Module):
    """ basic residual bottleneck

    Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    while original implementation places the stride at the first 1x1 convolution(self.conv1)
    according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    This variant is also known as ResNet V1.5 and improves accuracy according to
    https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    Args:
        inplanes (int): number of input channels
        planes (int): number of output channels
        stride (int): the stride for the convolution operation. Default: 1
        downsample (nn.Module): operations performed when residual is applied. Default: None
        groups (int): the groups for the convolution operation. Default: 1
        base_width (int): base width of image. Default: 64
        dilation (int): the dilation for the convolution operation. Default: 1
        norm_layer (nn.Module): normalization layer after convolution layer. Default: None
    """

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:

        super(ResidualBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
