import os
import random
from collections import OrderedDict
from os.path import basename, join

import cv2
import numpy as np

from gen_perlin import generate_perlin_noise
from gif import guided_filter
from load import load_pickle, load_hsi
from util import calculate_psnr_ssim, check_dtype, visualize_tool, normalize, get_random_image

from config import RAIN, LEVEL

BANDS = [136, 67, 18]
# BANDS = [36, 19, 8]


class RainModelForOurHSI:
    def __init__(self,
                 hsi_path=None,
                 streak_path=None,
                 bands_path=None,
                 r0=None,
                 level=None,
                 d=0.3,
                 a=1.0,
                 gif=False,
                 use_perlin=True,
                 alpha=1.0,
                 save_root_path=None
                 ):

        self.DV = {
            1: 20,
            2: random.uniform(15, 20),
            3: random.uniform(6, 8),
            4: random.uniform(3, 4),
        }

        self.hsi_path = hsi_path
        self.hsi = load_hsi(hsi_path)
        # 465索引波段似乎存在问题：全局NaN
        # self.hsi = np.delete(self.hsi, [465], axis=2)
        # self.hsi = normalize(self.hsi)
        self.streak_path = streak_path
        self.bands_path = bands_path
        self.r0 = random.uniform(0, 0.66) if not r0 else r0
        self.level = random.uniform(1, 3) if not level else level
        self.d = np.full(self.hsi.shape, d)
        self.a = np.full(self.hsi.shape, a)
        self.gif = gif
        self.use_perlin = use_perlin
        self.alpha = alpha
        self.save_root_path = save_root_path

        self.height, self.width, self.channel = self.hsi.shape
        self.dv = self.DV.get(self.level)
        self.rain_speed = RAIN.get(self.level)
        self.label = LEVEL.get(self.level)
        self.streak_path = f'{self.streak_path}/{self.label}'
        self.lambdas = load_pickle(self.bands_path)['bands']

        self.perlin_noise, self.rain_streak, self.tau_rain, self.tau_fog = self._init_params()

        self.deg_streak = None
        self.deg_rain_img, self.deg_fog_img, self.deg_rain_fog_img = None, None, None
        self.deg_img = None
        self.psnr_value = None
        self.ssim_value = None

    def _init_params(self):
        # ------------------------------------------- Perlin Noise ------------------------------------------- #
        if self.use_perlin:
            perlin_noise = generate_perlin_noise(impl='noise',
                                                 height=self.height,
                                                 width=self.width,
                                                 base=random.randint(1, 1000))
        else:
            perlin_noise = np.ones((self.height, self.width), dtype=np.uint8) * 255
        perlin_noise = check_dtype(perlin_noise)

        if self.gif:
            perlin_noise = guided_filter(guide_image=cv2.cvtColor(self.hsi[:, :, BANDS], cv2.COLOR_BGR2GRAY),
                                         input_image=perlin_noise,
                                         radius=8,
                                         epsilon=0.1)

        # ------------------------------------------- Rain Streak ------------------------------------------- #
        rain_streak = check_dtype(get_random_image(self.streak_path))

        rain_streak = np.expand_dims(rain_streak, axis=-1)
        rain_streak = np.tile(rain_streak, (1, 1, self.channel))

        # ------------------------------------------- Transmission ------------------------------------------- #
        gamma_rain = self.r0 * self.rain_speed ** 0.66
        if isinstance(gamma_rain, float):
            gamma_rain = np.ones(self.hsi.shape) * gamma_rain

        gamma_fog = np.exp(1.144 - 0.0128 * self.dv - (0.368 + 0.0214 * self.dv) * np.log(self.lambdas / 1e3)) / self.dv
        if len(gamma_fog.shape) == 1:
            gamma_fog = np.tile(gamma_fog, (self.height, self.width, 1))

        # mask for streak and fog
        # gamma_rain = gamma_rain * perlin_noise[..., None]
        gamma_fog = gamma_fog * normalize(perlin_noise.astype(np.float32))[..., None]

        tau_rain = np.exp(-gamma_rain * self.d)
        tau_fog = np.exp(-gamma_fog * self.d)

        return perlin_noise, rain_streak, tau_rain, tau_fog

    def synthesize(self):
        self.deg_streak = self.rain_streak * self.alpha * self.tau_fog

        self.deg_rain_img = self.hsi * self.tau_rain + self.a * (1 - self.tau_rain)
        self.deg_fog_img = self.hsi * self.tau_fog + self.a * (1 - self.tau_fog)

        tau = self.tau_fog * self.tau_rain
        self.deg_rain_fog_img = self.hsi * tau + self.a * (1 - tau)

        self.deg_img = self.deg_rain_fog_img + self.deg_streak
        self.deg_img = np.clip(self.deg_img, 0, 1)
        self.cal_metric()

    def cal_metric(self):
        self.psnr_value, self.ssim_value = calculate_psnr_ssim(self.hsi, self.deg_img)
        # print(f"PSNR: {self.psnr_value:.2f}, SSIM: {self.ssim_value:.4f}")

    def visualize(self, RGB=True):
        data_dict = OrderedDict()
        data_dict['Origin image'] = self.hsi
        data_dict['Perlin noise'] = self.perlin_noise
        data_dict['Rain streak'] = self.rain_streak
        data_dict['Rain streak With Scattering'] = self.deg_streak
        data_dict['Image affected by rain scattering'] = self.deg_rain_img
        data_dict['Image affected by fog scattering'] = self.deg_fog_img
        data_dict['Image affected by (rain, fog) scattering'] = self.deg_rain_fog_img
        data_dict['Degraded image'] = self.deg_img

        visualize_tool(fig_size=(20, 10),
                       rows_cols=(2, 4),
                       data_dict=data_dict,
                       save_path=f'result/OurHSI/{self.label}.jpg',
                       RGB=RGB)

    def save(self, save_gt=False):
        filename, ext = basename(self.hsi_path).split('.')
        rain_file = f'{filename}.{ext}'

        rain_path = join(self.save_root_path, 'rain')
        rain_path = join(rain_path, self.label)
        os.makedirs(rain_path, exist_ok=True)
        np.save(join(rain_path, rain_file), self.deg_img.astype(np.float32))

        if save_gt:
            gt_path = join(self.save_root_path, 'gt')
            os.makedirs(gt_path, exist_ok=True)
            np.save(join(gt_path, filename), self.hsi.astype(np.float32))


if __name__ == '__main__':
    # PSNR: 10 15 20 25
    # alpha: 0.7 0.7 0.95 1.0
    # seed = 42
    # random.seed(seed)

    model = RainModelForOurHSI(
        hsi_path='/home/disk2/ZR/datasets/OurHSI/extra/gt/2_21_4_1.npy',
        streak_path='/home/disk2/ZR/datasets/OurHSI/streakV2',
        bands_path='/home/disk2/ZR/datasets/OurHSI/meta.pkl',
        r0=0.248,  # 0.248
        level=2,
        a=1,
        d=0.3,
        gif=True,
        alpha=0.75
    )

    model.synthesize()
    model.visualize(RGB=True)
