import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss, L1Loss, CrossEntropyLoss

from pytorch_wavelets import DWTForward, DWTInverse

from .builder import CRITERION

CRITERION.register_module(module=MSELoss)
CRITERION.register_module(module=L1Loss)
CRITERION.register_module(module=CrossEntropyLoss)


@CRITERION.register_module()
class MultiL1Loss(nn.Module):
    """ multi-branch L1Loss

    When the network has multiple pipelines, the loss is calculated for
    each pipeline and then these losses are weighted and constitute the final loss.

    Args:
        weights (List[int] | int): weights of each pipeline
            If a list is provided, it will be weighted according to the value of the list,
            if not, the mean weight will be assigned initially and updated with backward propagation.
            Default: 2
    """

    def __init__(self,
                 weights=2):
        super(MultiL1Loss, self).__init__()
        self.weights = weights

    def forward(self, prediction, target):
        loss = 0
        for idx, pred in enumerate(prediction):
            loss += F.l1_loss(pred, target) * self.weights[idx]

        return loss


@CRITERION.register_module()
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): 模型输出，形状为 (B, C, H, W)
            target (torch.Tensor): 真实标签，形状为 (B, C, H, W)

        Returns:
            loss (torch.Tensor): 标量损失值
        """
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps).mean()
        return loss


class SpectrumConstancyLoss(nn.Module):
    def __init__(self):
        super(SpectrumConstancyLoss, self).__init__()
        # 定义差分卷积核（上下左右）
        self.register_buffer('diff_kernel', torch.tensor([[-1, 0, 1]]).float())  # 3x1 差分核

    def forward(self, pred):
        """
        Args:
            pred (torch.Tensor): 输入图像，形状为 (B, C, H, W)

        Returns:
            loss (torch.Tensor): 标量，Spectrum Constancy Loss 值
        """
        B, C, H, W = pred.shape

        # 计算水平方向差分（dx）
        dx = F.conv2d(pred, weight=self.diff_kernel.view(1, 1, 1, 3).repeat(C, 1, 1, 1),
                      padding=(0, 1), groups=C)  # (B, C, H, W)

        # 计算垂直方向差分（dy）
        dy = F.conv2d(pred, weight=self.diff_kernel.view(1, 1, 3, 1).repeat(C, 1, 1, 1),
                      padding=(1, 0), groups=C)  # (B, C, H, W)

        # 在通道维度上求平方和（欧氏距离平方）
        dx_sq = dx.pow(2).sum(dim=1)  # (B, H, W)
        dy_sq = dy.pow(2).sum(dim=1)  # (B, H, W)

        # 求总损失（平均或求和）
        loss = (dx_sq + dy_sq).mean()

        return loss


@CRITERION.register_module()
class AdaptiveSpatialSpectralLossV1(nn.Module):
    def __init__(self, eps=1e-3, min_alpha=0.1, max_alpha=0.9):
        """
        Args:
            eps: Charbonnier 平滑系数
            min_alpha/max_alpha: 动态alpha的裁剪范围（避免极端情况）
        """
        super().__init__()
        self.eps = eps
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

    def forward(self, pred, target):
        diff = torch.abs(pred - target)  # [B, C, H, W]

        # 计算空间损失 [B, C, H, W] -> 标量
        spatial_loss = torch.sqrt(diff ** 2 + self.eps ** 2).mean()  # L1

        # 计算光谱损失 [B, H, W] -> 标量
        spectral_loss = torch.sqrt(torch.mean(diff ** 2, dim=1) + self.eps ** 2).mean()  # L2

        # 动态alpha（带裁剪）
        alpha = spatial_loss / (spatial_loss + spectral_loss + 1e-6)
        alpha = torch.clamp(alpha, self.min_alpha, self.max_alpha)

        # 联合损失
        joint_loss = alpha * spatial_loss + (1 - alpha) * spectral_loss

        return joint_loss


@CRITERION.register_module()
class AdaptiveSpatialSpectralLossV2(nn.Module):
    def __init__(self, eps=1e-3, min_alpha=0.1, max_alpha=0.9):
        """
        Args:
            eps: Charbonnier 平滑系数
            min_alpha/max_alpha: 动态alpha的裁剪范围（避免极端情况）
        """
        super().__init__()
        self.eps = eps
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

    def forward(self, pred, target):
        B, C, H, W = pred.shape
        diff = torch.abs(pred - target)  # [B, C, H, W]

        spatial_diff = torch.mean(diff.view(B, C, -1), dim=1)  # B L
        spectral_diff = torch.mean(diff.view(B, C, -1), dim=2)  # B C

        spatial_loss = torch.sqrt(spatial_diff ** 2 + self.eps ** 2).mean()  # L1
        spectral_loss = torch.sqrt(spectral_diff ** 2 + self.eps ** 2).mean()  # L2

        # 动态alpha（带裁剪）
        alpha = spatial_loss / (spatial_loss + spectral_loss + 1e-6)
        alpha = torch.clamp(alpha, self.min_alpha, self.max_alpha)

        # 联合损失
        joint_loss = alpha * spatial_loss + (1 - alpha) * spectral_loss

        return joint_loss


class WaveletProcessor(nn.Module):
    def __init__(self, wavelet='haar', level=2, mode='reflect'):
        super().__init__()
        self.dwt = DWTForward(wave=wavelet, J=level, mode=mode)  # 小波分解
        self.idwt = DWTInverse(wave=wavelet)  # 小波重构
        self.level = level

    def forward(self, x):
        """输入x: [B, C, H, W]"""
        # 小波分解
        coeffs = self.dwt(x)  # 返回 (LL, [LH, HL, HH] * level)

        # 提取各频带分量
        LL = coeffs[0]  # 低频分量 [B, C, H/(2^level), W/(2^level)]
        HL = [c[0][:, :, 0] for c in coeffs[1]]  # 水平高频
        LH = [c[0][:, :, 1] for c in coeffs[1]]  # 垂直高频
        HH = [c[0][:, :, 2] for c in coeffs[1]]  # 对角线高频

        return {
            'LL': LL,  # 低频
            'HL': HL,  # 水平高频（边缘细节）
            'LH': LH,  # 垂直高频（边缘细节）
            'HH': HH  # 对角线高频（纹理）
        }


@CRITERION.register_module()
class SpatialSpectralFreqLoss(nn.Module):
    def __init__(self, version='v1', eps=1e-3, wave='haar', J=2, mode='reflect'):
        """
        Args:
            wave: 小波基类型 ('haar', 'db2'等)
            J: 小波分解层数
        """
        super().__init__()
        self.version = version
        self.eps = eps
        self.wavelet = WaveletProcessor(wave, J, mode).cuda()
        self.freq_loss = CharbonnierLoss(eps=eps)

    def forward(self, pred, target):
        B, C, H, W = pred.shape
        diff = torch.abs(pred - target)
        if self.version == 'v1':
            spatial_loss = torch.sqrt(diff ** 2 + self.eps ** 2).mean()  # L1
            spectral_loss = torch.sqrt(torch.mean(diff ** 2, dim=1) + self.eps ** 2).mean()
        elif self.version == 'v2':
            spatial_diff = torch.mean(diff.view(B, C, -1), dim=1)  # B L
            spectral_diff = torch.mean(diff.view(B, C, -1), dim=2)  # B C
            spatial_loss = torch.sqrt(spatial_diff ** 2 + self.eps ** 2).mean()  # L1
            spectral_loss = torch.sqrt(spectral_diff ** 2 + self.eps ** 2).mean()  # L2
        else:
            raise NotImplementedError(f'version:{self.version} not supported yet !')

        # 小波频域损失
        pred_wave = self.wavelet(pred)  # 预测图像的小波系数
        target_wave = self.wavelet(target)  # 真实图像的小波系数
        freq_loss = 0
        weights = [0.5, 0.2, 0.2, 0.1]
        keys = ['LL', 'HL', 'LH', 'HH']
        for key, weight in zip(keys, weights):
            for p, t in zip(pred_wave[key], target_wave[key]):
                freq_loss += self.freq_loss(p, t) * weight

        # 加权联合损失
        alpha = spatial_loss / (spatial_loss + spectral_loss + freq_loss + 1e-6)
        beta = spectral_loss / (spatial_loss + spectral_loss + freq_loss + 1e-6)
        total_loss = alpha * spatial_loss + beta * spectral_loss + (1 - alpha - beta) * freq_loss
        return total_loss


if __name__ == '__main__':
    _pred = torch.randn(1, 224, 128, 128)
    criterion = SpectrumConstancyLoss()
    criterion(_pred)
