import torch
from pytorch_wavelets import DWTForward, DWT1DForward
from torch import nn

from uchiha.models.modules.common import sequence_to_image
from uchiha.models.builder import MODULE


@MODULE.register_module()
class DWT2d(nn.Module):
    """ 2D Discrete Wavelet Transform

    Args:
        scales (int): Number of scales (number of transform performed)
        wave (str): The type of wavelet
        padding (str): data padding mode before transformation
    """

    def __init__(self,
                 scales=1,
                 wave='haar',
                 padding='zero'):
        super().__init__()
        self.wavelet_transform = DWTForward(J=scales, wave=wave, mode=padding)

    def forward(self, x):
        # x.shape:B, C, H, W
        out = self.wavelet_transform(x)
        LL, H = out
        LH, HL, HH = H[0][:, :, 0, :, :], H[0][:, :, 1, :, :], H[0][:, :, 2, :, :]
        return torch.cat((LL, LH, HL, HH), dim=1)


@MODULE.register_module()
class DWT1d(nn.Module):
    """ 1D Discrete Wavelet Transform

    Args:
        scales (int): Number of scales (number of transform performed)
        wave (str): The type of wavelet
        padding (str): data padding mode before transformation
        origin (bool): Whether to return original data.
    """

    def __init__(self,
                 scales=1,
                 wave='haar',
                 padding='zero',
                 origin=False):
        super().__init__()
        self.wavelet_transform = DWT1DForward(J=scales, wave=wave, mode=padding)
        self.origin = origin

    def forward(self, x):
        # x.shape:B, C, H, W
        if len(x.shape) == 4:
            x = x.flatten(2).transpose(1, 2)

        L, H = self.wavelet_transform(x)

        parallel = [sequence_to_image(L)]
        for idx, h in enumerate(H):
            parallel.append(sequence_to_image(h))

        if self.origin:
            return x, parallel
        else:
            return parallel


@MODULE.register_module()
class PreSimpleConv(nn.Module):
    def __init__(self,
                 in_channel=3,
                 out_channel=64,
                 depth=2,
                 stride=1,
                 copies=1):
        super().__init__()
        self.depth = depth

        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(in_channel, out_channel, 3, stride, 1))
        for i in range(1, self.depth):
            self.convs.append(nn.Conv2d(out_channel, out_channel, 3, 1, 1))

        self.bns = nn.ModuleList([nn.BatchNorm2d(out_channel) for _ in range(self.depth)])
        self.activate = nn.ReLU(inplace=True)

        self.copies = copies

    def forward(self, x):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x)
            x = bn(x)
            x = self.activate(x)
        return x if self.copies == 1 else [x for _ in range(self.copies)]


def get_serial_rcp(hyperspectral_tensor, split_bands):
    B, C, H, W = hyperspectral_tensor.shape
    if split_bands is None:
        split_bands = [C // 2]
    rcps = []

    for split_band in split_bands:
        # 1. 分组处理 (沿通道维度)
        heavy = hyperspectral_tensor[:, :split_band, :, :]  # B×split_band×H×W
        light = hyperspectral_tensor[:, split_band:, :, :]  # B×(C-split_band)×H×W

        # 2. 计算组内均值 (沿通道维度)
        heavy_mean = torch.mean(heavy, dim=1, keepdim=True)  # B×1×H×W
        light_mean = torch.mean(light, dim=1, keepdim=True)  # B×1×H×W

        # 3. 计算差值并归一化
        rcp = light_mean - heavy_mean
        rcp_min = rcp.view(B, -1).min(dim=1)[0].view(B, 1, 1, 1)
        rcp_max = rcp.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
        rcp = (rcp - rcp_min) / (rcp_max - rcp_min + 1e-8)

        rcps.append(rcp)

    return torch.cat(rcps, dim=1)  # B×L×H×W


def get_sorted_rcp(hyperspectral_tensor, split_bands=None):
    B, C, H, W = hyperspectral_tensor.shape
    if split_bands is None:
        split_bands = [C // 2]
    rcps = []

    # 1. 计算每个通道的全局均值 [B, C]
    channel_means = torch.mean(hyperspectral_tensor.reshape(B, C, -1), dim=2)  # B×C

    # 2. 按均值排序并分组 (每个样本独立排序)
    sorted_indices = torch.argsort(channel_means, dim=1)  # B×C

    for split_band in split_bands:
        # 3. 计算组内均值差
        # 获取分组索引 (B×split_band) 和 (B×(C-split_band))
        low_group = sorted_indices[:, :split_band]  # B×split_band
        high_group = sorted_indices[:, split_band:]  # B×(C-split_band)

        # 收集低/高组的像素值 (使用 gather 按索引选择通道)
        # 扩展索引维度以匹配输入形状
        low_values = torch.gather(
            hyperspectral_tensor,
            dim=1,
            index=low_group.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        )  # B×split_band×H×W

        high_values = torch.gather(
            hyperspectral_tensor,
            dim=1,
            index=high_group.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        )  # B×(C-split_band)×H×W

        # 计算组内均值 (沿通道维度)
        low_mean = torch.mean(low_values, dim=1, keepdim=True)  # B×1×H×W
        high_mean = torch.mean(high_values, dim=1, keepdim=True)  # B×1×H×W
        rcp_image = high_mean - low_mean  # B×1×H×W

        # 4. 归一化 (每个样本独立归一化)
        rcp_min = rcp_image.reshape(B, -1).min(dim=1)[0].view(B, 1, 1, 1)
        rcp_max = rcp_image.reshape(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
        rcp_image = (rcp_image - rcp_min) / (rcp_max - rcp_min + 1e-8)

        rcps.append(rcp_image)  # 添加到结果列表

    return torch.cat(rcps, dim=1)  # B×L×H×W


def get_sorted_rcp_pw(image: torch.Tensor, split_bands: list) -> torch.Tensor:
    """
    输入: CHW torch.Tensor
    功能: 排序 + 切分点差值
    输出: HWN torch.Tensor
    """
    c, h, w = image.shape
    flattened = image.permute(1, 2, 0).reshape(-1, c)  # L x C
    sorted_flattened, _ = torch.sort(flattened, dim=1)

    results = []
    for split_idx in split_bands:
        if not (0 < split_idx < c):
            raise ValueError(f"切分点 {split_idx} 超出通道范围 (1~{c - 1})")
        front = sorted_flattened[:, :split_idx]
        back = sorted_flattened[:, split_idx:]
        diff = front.mean(dim=1) - back.mean(dim=1)
        results.append(diff)

    output = torch.stack(results, dim=1)  # L x N
    return output.reshape(h, w, -1)  # HWN


@MODULE.register_module()
class GroupRCP(nn.Module):
    """ Group-based Residue Channel Prior

    Args:
        split_bands (List): 划分的波段索引，在其两侧进行残差运算
    """

    def __init__(self, split_bands=None, strategy=None):
        super().__init__()
        self.split_bands = split_bands

        if strategy is None:
            self.strategy = ['sorted', 'serial']
        else:
            self.strategy = strategy

    def forward(self, x):
        # x.shape (B, C, H, W)
        rcp = []
        for s in self.strategy:
            if s == 'sorted':
                rcp.append(get_sorted_rcp(x, self.split_bands))
            elif s == 'serial':
                rcp.append(get_serial_rcp(x, self.split_bands))
            elif s == 'sorted_pw':
                rcp.append(get_sorted_rcp_pw(x, self.split_bands))
            else:
                raise ValueError(f"Invalid strategy: {s}")
        rcp = torch.cat(rcp, dim=1)

        return rcp


if __name__ == '__main__':
    dwt = DWT1DForward(wave='db6', J=3)
    X = torch.randn(10, 5, 100)
    yl, yh = dwt(X)
