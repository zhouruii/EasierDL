import warnings

import numpy as np
import torch

from ..builder import PIPELINES
from ...utils import impad, impad_to_multiple, img_normalize


@PIPELINES.register_module()
class Pad:
    """Pad the image.

    There are two padding modes:
    (1) pad to a fixed size
    (2) pad to the minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (dict, optional): A dict for padding value, the default
            value is `0`.
    """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_val=0,
                 mode='constant'):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.mode = mode

        assert size is not None or size_divisor is not None, \
            'only one of size and size_divisor should be valid'

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        pad_val = self.pad_val
        if 'sample' in results:
            key = 'sample'
        else:
            key = None  # TODO 数据的键名泛用性补充

        results['pad_cfg'] = dict(ori_shape=results[key].shape, pad_shape=None)

        if self.size is not None:
            padded_img = impad(
                results[key], shape=self.size, pad_val=pad_val, padding_mode=self.mode)
        elif self.size_divisor is not None:
            padded_img = impad_to_multiple(
                results[key], self.size_divisor, pad_val=pad_val, padding_mode=self.mode)
        results[key] = padded_img
        results['pad_cfg']['pad_shape'] = padded_img.shape

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class Normalize:
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean=None, std=None, to_rgb=True, scope='spatial', mode='standard'):

        self.mean = np.array(mean, dtype=np.float32) if mean else None
        self.std = np.array(std, dtype=np.float32) if std else None
        self.to_rgb = to_rgb
        self.scope = scope
        self.mode = mode

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results['sample'] = img_normalize(results['sample'], self.mean, self.std,
                                     self.to_rgb, self.scope, self.mode)
        if self.mean and self.std:
            results['norm_cfg'] = dict(
                mean=self.mean, std=self.std, to_rgb=self.to_rgb, scope=self.scope)
        else:
            results['norm_cfg'] = dict(
                mean='auto', std='auto', to_rgb=self.to_rgb, scope=self.scope)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class EasyToTensor:
    def __init__(self, mode='CHW'):
        self.mode = mode

    def __call__(self, results):
        assert 'sample' in results and 'target' in results, 'sample and target should be loaded to results'

        if self.mode == 'CHW':
            results['sample'] = torch.Tensor(results['sample']).permute(2, 0, 1)
        else:
            results['sample'] = torch.Tensor(results['sample'])

        results['target'] = torch.Tensor(results['target'])

        return results
