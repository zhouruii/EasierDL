import random

import cv2

from ..builder import PIPELINES


@PIPELINES.register_module()
class RandomCrop:
    def __init__(self, crop_size, prob=0.5):
        """
        Args:
            crop_size (tuple): 裁剪的目标尺寸 (height, width)
            prob (float): 触发增强的概率
        """
        if isinstance(crop_size, int):
            crop_size = [crop_size, crop_size]
        self.size = crop_size
        self.p = prob

    def __call__(self, data):
        if random.random() < self.p:
            sample = data['sample']
            target = data['target']

            h, w = sample.shape[:2]
            th, tw = self.size

            # 如果图像比裁剪尺寸小，则先 resize 到裁剪尺寸
            if h < th or w < tw:
                sample = cv2.resize(sample, (tw, th), interpolation=cv2.INTER_LINEAR)
                target = cv2.resize(target, (tw, th), interpolation=cv2.INTER_NEAREST)
                data['sample'] = sample
                data['target'] = target
                return data

            # 随机选择裁剪起点
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)

            # 裁剪图像
            if len(sample.shape) == 3:
                sample_cropped = sample[i:i + th, j:j + tw, :]
            else:
                sample_cropped = sample[i:i + th, j:j + tw]

            if len(target.shape) == 3:
                target_cropped = target[i:i + th, j:j + tw, :]
            else:
                target_cropped = target[i:i + th, j:j + tw]

            data['sample'] = sample_cropped
            data['target'] = target_cropped

        return data


@PIPELINES.register_module()
class CenterCrop:
    def __init__(self, size):
        """
        Args:
            size (tuple): 裁剪的目标尺寸 (height, width)
        """
        self.size = size

    def __call__(self, data):
        sample = data['sample']
        target = data['target']

        h, w = sample.shape[:2]
        th, tw = self.size

        # 如果图像比裁剪尺寸小，则先 resize 到裁剪尺寸
        if h < th or w < tw:
            import cv2
            sample = cv2.resize(sample, (tw, th), interpolation=cv2.INTER_LINEAR)
            target = cv2.resize(target, (tw, th), interpolation=cv2.INTER_NEAREST)
            data['sample'] = sample
            data['target'] = target
            return data

        # 计算中心裁剪的起始坐标
        i = (h - th) // 2
        j = (w - tw) // 2

        # 执行裁剪
        if len(sample.shape) == 3:
            sample_cropped = sample[i:i + th, j:j + tw, :]
        else:
            sample_cropped = sample[i:i + th, j:j + tw]

        if len(target.shape) == 3:
            target_cropped = target[i:i + th, j:j + tw, :]
        else:
            target_cropped = target[i:i + th, j:j + tw]

        data['sample'] = sample_cropped
        data['target'] = target_cropped

        return data
