import logging
import os
from os.path import join

import numpy as np
from colorlog import ColoredFormatter

from model_aviris import RainModelForAVIRIS
from config import LEVEL
from model_ourhsi import RainModelForOurHSI


# 配置日志
def setup_logger(log_file='psnr_ssim_log.txt'):
    """
    配置日志记录器。

    Args:
        log_file (str): 日志文件的路径。
    """
    # 创建日志记录器
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)

    # 控制台处理器
    console_handler = logging.StreamHandler()

    # 控制台日志颜色格式
    console_formatter = ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(message)s",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )

    # 应用控制台格式
    console_handler.setFormatter(console_formatter)

    # 文件处理器
    file_handler = logging.FileHandler(log_file, mode="w")

    # 文件日志格式
    # file_formatter = logging.Formatter("%(asctime)s - %(levelname)-8s - %(message)s")
    # file_handler.setFormatter(file_formatter)

    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


if __name__ == '__main__':
    ALPHA = [0.6, 0.7, 0.8, 0.9]
    hsi_root_path = '/home/disk2/ZR/datasets/AVIRIS/512/gt'
    filenames = os.listdir(hsi_root_path)
    hsi_paths = [join(hsi_root_path, file) for file in filenames]

    logger = setup_logger('/home/disk2/ZR/datasets/AVIRIS/512/synthesis.log')

    for idx, hsi_path in enumerate(hsi_paths):
        for level in range(1, 5):
            model = RainModelForAVIRIS(
                hsi_path=hsi_path,
                bands_path='/home/disk2/ZR/datasets/AVIRIS/512/meta.pkl',
                r0=0.155,
                level=level,
                a=1,
                d=0.4,
                gif=True,
                alpha=ALPHA[level - 1],
                save_root_path='/home/disk2/ZR/datasets/AVIRIS/512'
            )
            model.synthesize()
            model.save()
            log_info = (f'Processed:[{idx + 1} / {len(hsi_paths)}] \t File: {filenames[idx]} \t '
                        f'Level: {LEVEL.get(level)} \t PSNR: {model.psnr_value:.2f} \t SSIM: {model.ssim_value:.4f}')
            if np.isnan(model.psnr_value) or np.isnan(model.ssim_value):
                logger.error(log_info)
            else:
                logger.info(log_info)

    # ALPHA = [0.7, 0.75, 0.8, 0.9]
    # hsi_root_path = '/home/disk2/ZR/datasets/OurHSI/extra/gt'
    # filenames = os.listdir(hsi_root_path)
    # hsi_paths = [join(hsi_root_path, file) for file in filenames]
    #
    # logger = setup_logger('/home/disk2/ZR/datasets/OurHSI/extra/synthesis.log')
    #
    # for idx, hsi_path in enumerate(hsi_paths):
    #     for level in range(1, 5):
    #         model = RainModelForOurHSI(
    #             hsi_path=hsi_path,
    #             streak_path='/home/disk2/ZR/datasets/OurHSI/streakV2',
    #             bands_path='/home/disk2/ZR/datasets/OurHSI/meta.pkl',
    #             r0=0.248,
    #             level=level,
    #             a=1,
    #             d=0.3,
    #             gif=True,
    #             alpha=ALPHA[level - 1],
    #             save_root_path='/home/disk2/ZR/datasets/OurHSI/extra'
    #         )
    #         model.synthesize()
    #         model.save()
    #         log_info = (f'Processed:[{idx + 1} / {len(hsi_paths)}] \t File: {filenames[idx]} \t '
    #                     f'Level: {LEVEL.get(level)} \t PSNR: {model.psnr_value:.2f} \t SSIM: {model.ssim_value:.4f}')
    #         if np.isnan(model.psnr_value) or np.isnan(model.ssim_value):
    #             logger.error(log_info)
    #         else:
    #             logger.info(log_info)
