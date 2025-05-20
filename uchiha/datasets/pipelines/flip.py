import random

from ..builder import PIPELINES


@PIPELINES.register_module()
class RandomFlip:
    def __init__(self, probs=None, directions=None):
        """
        Args:
            probs (list): 四种翻转的概率权重，顺序为：
                      [原图概率, 水平翻转概率, 垂直翻转概率, 对角翻转概率]
        """
        if directions is None:
            directions = ['original', 'horizontal', 'vertical', 'diagonal']
        if probs is None:
            probs = [0.5, 0.25, 0.25, 0.0]  # 默认设置

        self.p = probs

        # 归一化概率
        total = sum(probs)
        self.norm_p = [x / total for x in probs]

        # 构建选择空间
        self.directions = directions

        # 累积概率用于随机选择
        cum = 0.0
        self.cum_probs = []
        for probs in self.norm_p:
            cum += probs
            self.cum_probs.append(cum)

    def _apply_flip(self, img, flip_type):
        """执行具体的翻转操作"""
        if flip_type == 'original':
            return img
        elif flip_type == 'horizontal':
            return img[:, ::-1, ...].copy()  # HWC 或 HW
        elif flip_type == 'vertical':
            return img[::-1, :, ...].copy()
        elif flip_type == 'diagonal':
            return img[:, ::-1, ...][::-1, :, ...].copy()
        else:
            raise ValueError(f"Unknown flip type: {flip_type}")

    def __call__(self, data):
        sample = data['sample']
        target = data['target']

        # 使用累积概率选择翻转类型
        rand_val = random.random()
        selected = self.directions[-1]  # 默认选最后一个
        for i, cp in enumerate(self.cum_probs):
            if rand_val < cp:
                selected = self.directions[i]
                break

        # 应用翻转
        sample = self._apply_flip(sample, selected)
        target = self._apply_flip(target, selected)

        data['sample'] = sample
        data['target'] = target

        return data
