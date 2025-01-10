import random
from collections import OrderedDict

import cv2
import numpy as np

from tools.synthesis.gen_perlin import generate_perlin_noise
from tools.synthesis.gen_streak import generate_bird_view_streak
from tools.synthesis.gif import guided_filter
from tools.synthesis.util import read_img, calculate_psnr_ssim, check_dtype, visualize_tool

DV = {
    1: random.uniform(10, 20),  # 小雨
    2: random.uniform(4, 10),  # 中雨
    3: random.uniform(2, 4),  # 大雨
    4: random.uniform(1, 2),  # 暴雨
}

RAIN = {
    1: 0.2083,  # 小雨
    2: 0.829167,  # 中雨
    3: 1.87083,  # 大雨
    4: 4.1625,  # 暴雨
}

RAIN_STREAK = {
    1: dict(num_drops=random.randint(2000, 2500), streak_length=random.randint(20, 25),
            wind_angle=random.randint(-180, 180), wind_strength=random.uniform(0, 0.05)),  # 小雨
    2: dict(num_drops=random.randint(3000, 3500), streak_length=random.randint(30, 35),
            wind_angle=random.randint(-180, 180), wind_strength=random.uniform(0.25, 0.3)),  # 中雨
    3: dict(num_drops=random.randint(4000, 4500), streak_length=random.randint(40, 45),
            wind_angle=random.randint(-180, 180), wind_strength=random.uniform(0, 0.09)),  # 大雨
    4: dict(num_drops=random.randint(5000, 5500), streak_length=random.randint(50, 55),
            wind_angle=random.randint(-180, 180), wind_strength=random.uniform(0, 0.12)),  # 暴雨
}


class RainModel:
    def __init__(self,
                 img_path=None,
                 lambdas=None,
                 r0=None,
                 level=None,
                 d=1.0,
                 a=1.0,
                 scales=None,
                 gif=False,
                 depth=None,
                 use_perlin=True,
                 ):
        self.img = read_img(img_path)
        self.lambdas = np.array([700, 540, 438]) if not lambdas else lambdas  # RGB
        self.r0 = random.uniform(0, 0.66) if not r0 else r0
        self.level = random.uniform(1, 4) if not level else level
        self.d = np.full(self.img.shape, d)
        self.a = np.full(self.img.shape, a)
        self.scales = scales
        self.gif = gif
        self.use_perlin = use_perlin

        self.height, self.width, self.channel = self.img.shape
        self.depth = depth if depth is not None else (self.height + self.width) // 2
        self.dv = DV.get(self.level)
        self.rain_speed = RAIN.get(self.level)

        self.perlin_noise, self.rain_streak, self.tau_rain, self.tau_fog = self._init_params()

        self.deg_streak = None
        self.deg_rain_img, self.deg_fog_img, self.deg_rain_fog_img = None, None, None
        self.deg_img = None

    def _init_params(self):
        # ------------------------------------------- Perlin Noise ------------------------------------------- #
        if self.use_perlin:
            perlin_noise = generate_perlin_noise(impl='noise',
                                                 height=self.height,
                                                 width=self.width,
                                                 scales=self.scales)
            if self.gif:
                perlin_noise = guided_filter(guide_image=cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY),
                                             input_image=perlin_noise,
                                             radius=None,
                                             epsilon=None)
        else:
            perlin_noise = np.ones((self.height, self.width), dtype=np.uint8) * 255
        perlin_noise = check_dtype(perlin_noise)

        # ------------------------------------------- Rain Streak ------------------------------------------- #
        rain_streak = generate_bird_view_streak(height=self.height,
                                                width=self.width,
                                                depth=self.depth,
                                                # noise=perlin_noise,
                                                **RAIN_STREAK.get(self.level))
        rain_streak = check_dtype(rain_streak)

        if len(rain_streak.shape) < 3:
            rain_streak = np.expand_dims(rain_streak, axis=-1)
            rain_streak = np.tile(rain_streak, (1, 1, self.channel))

        # ------------------------------------------- Transmission ------------------------------------------- #
        gamma_rain = self.r0 * self.rain_speed ** 0.66
        if isinstance(gamma_rain, float):
            gamma_rain = np.ones(self.img.shape) * gamma_rain

        gamma_fog = np.exp(1.144 - 0.0128 * self.dv - (0.368 + 0.0214 * self.dv) * np.log(self.lambdas / 1e3)) / self.dv
        if len(gamma_fog.shape) == 1:
            gamma_fog = np.tile(gamma_fog, (self.height, self.width, 1))

        gamma_fog = gamma_fog * perlin_noise[..., None]

        tau_rain = np.exp(-gamma_rain * self.d)
        tau_fog = np.exp(-gamma_fog * self.d)

        return perlin_noise, rain_streak, tau_rain, tau_fog

    def synthesize(self):
        self.deg_streak = self.rain_streak * self.tau_fog

        self.deg_rain_img = self.img * self.tau_rain + self.a * (1 - self.tau_rain)
        self.deg_fog_img = self.img * self.tau_fog + self.a * (1 - self.tau_fog)

        tau = self.tau_fog * self.tau_rain
        self.deg_rain_fog_img = self.img * tau + self.a * (1 - tau)

        self.deg_img = self.deg_rain_fog_img + self.deg_streak
        self.cal_metric()

    def cal_metric(self):
        psnr_value, ssim_value = calculate_psnr_ssim(self.img, self.deg_img)
        print(f"PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}")

    def visualize(self):
        data_dict = OrderedDict()
        data_dict['Origin image'] = self.img
        data_dict['Perlin noise'] = self.perlin_noise
        data_dict['Rain streak'] = self.rain_streak
        data_dict['Rain streak With Scattering'] = self.deg_streak
        data_dict['Image affected by rain scattering'] = self.deg_rain_img
        data_dict['Image affected by fog scattering'] = self.deg_fog_img
        data_dict['Image affected by (rain, fog) scattering'] = self.deg_rain_fog_img
        data_dict['Degraded image'] = self.deg_img

        visualize_tool(fig_size=(20, 10),
                       rows_cols=(2, 4),
                       data_dict=data_dict)


if __name__ == '__main__':
    model = RainModel(
        img_path='demo/5.jpg',
        r0=0.248,
        level=2,
        a=1,
        d=1,
        scales=[300]
    )

    model.synthesize()
    model.visualize()
