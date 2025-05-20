import random

import cv2

from ..builder import PIPELINES


@PIPELINES.register_module()
class RandomRotation:
    def __init__(self, angles=None, prob=0.5, interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_REFLECT_101):
        """
        Args:
            angles (tuple): 旋转角度范围，例如 (-30, 30)
            prob (float): 触发增强的概率
            interpolation (int): 插值方法（cv2.INTER_NEAREST / INTER_LINEAR / INTER_CUBIC）
            border_mode (int): 边界填充方式（cv2.BORDER_CONSTANT / BORDER_REPLICATE / BORDER_REFLECT_101）
        """
        if angles is None:
            angles = [-90, 90]
        self.angles = angles
        self.p = prob
        self.interpolation = interpolation
        self.border_mode = border_mode

    def __call__(self, data):
        if random.random() < self.p:
            sample = data['sample']
            target = data['target']

            h, w = sample.shape[:2]
            center = (w / 2, h / 2)

            # 随机生成旋转角度
            angle = random.choice(self.angles)
            if isinstance(angle, int):
                angle = float(angle)

            # 计算旋转矩阵
            M = cv2.getRotationMatrix2D(center, angle, scale=1.0)

            # 应用仿射变换
            if len(sample.shape) == 3:  # HWC
                sample_rotated = cv2.warpAffine(sample, M, (w, h), flags=self.interpolation,
                                                borderMode=self.border_mode)
            else:  # HW
                sample_rotated = cv2.warpAffine(sample, M, (w, h), flags=self.interpolation,
                                                borderMode=self.border_mode)

            if len(target.shape) == 3:  # HWC
                target_rotated = cv2.warpAffine(target, M, (w, h), flags=self.interpolation,
                                                borderMode=self.border_mode)
            else:  # HW
                target_rotated = cv2.warpAffine(target, M, (w, h), flags=self.interpolation,
                                                borderMode=self.border_mode)

            data['sample'] = sample_rotated
            data['target'] = target_rotated

        return data
