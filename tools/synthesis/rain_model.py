import random

import cv2
import numpy as np
from matplotlib import pyplot as plt

from tools.synthesis.gen_perlin import generate_perlin_noise
from tools.synthesis.gen_streak import generate_rain_streak
from tools.synthesis.gif import guided_filter
from tools.synthesis.util import read_img, to_visualize, calculate_psnr_ssim

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
    1: dict(num_drops=random.randint(1000, 1200), drop_length=random.randint(10, 12),
            angle=random.randint(5, 10), intensity=0.5),  # 小雨
    2: dict(num_drops=random.randint(1200, 1500), drop_length=random.randint(12, 15),
            angle=random.randint(10, 15), intensity=0.6),  # 中雨
    3: dict(num_drops=random.randint(1500, 2000), drop_length=random.randint(15, 18),
            angle=random.randint(15, 20), intensity=0.7),  # 大雨
    4: dict(num_drops=random.randint(2000, 2500), drop_length=random.randint(18, 20),
            angle=random.randint(20, 30), intensity=0.8),  # 暴雨
}


class RainModel:
    def __init__(self,
                 img_path=None,
                 lambdas=None,
                 r0=None,
                 level=None,
                 d=1.0,
                 a=1.0,
                 scale=45,
                 gif=False,
                 ):
        self.img = read_img(img_path)
        self.lambdas = np.array([700, 540, 438]) if not lambdas else lambdas  # RGB
        self.r0 = random.uniform(0, 0.66) if not r0 else r0
        self.level = random.uniform(1, 4) if not level else level
        self.d = np.full(self.img.shape, d)
        self.a = np.full(self.img.shape, a)
        self.scales = scale
        self.gif = gif

        self.height, self.width, self.channel = self.img.shape
        self.dv = DV.get(self.level)
        self.rain_speed = RAIN.get(self.level)
        self.tau_rain, self.tau_fog = self.cal_trans()
        self.rain_streak = self.gen_rain_streak()

        self._init_params()

        self.deg = None

    def _init_params(self):
        perlin_noise = generate_perlin_noise(impl='noise', height=self.height, width=self.width, scales=self.scales)
        if self.gif:
            perlin_noise = guided_filter(guide_image=cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY),
                                         input_image=perlin_noise,
                                         radius=None,
                                         epsilon=None)
        pass

    def gen_rain_streak(self):
        rain_streak = generate_rain_streak(
            self.height, self.width, **RAIN_STREAK.get(self.level))

        if len(rain_streak.shape) < 3:
            rain_streak = np.expand_dims(rain_streak, axis=-1)
            rain_streak = np.tile(rain_streak, (1, 1, self.channel))

        return rain_streak

    def cal_trans(self):
        gamma_rain = self.r0 * self.rain_speed
        if isinstance(gamma_rain, float):
            gamma_rain = np.ones(self.img.shape) * gamma_rain

        gamma_fog = np.exp(1.144 - 0.0128 * self.dv - (0.368 + 0.0214 * self.dv) * np.log(self.lambdas / 1e3)) / self.dv
        if len(gamma_fog.shape) == 1:
            gamma_fog = np.tile(gamma_fog, (self.height, self.width, 1))

        tau_rain = np.exp(-gamma_rain * self.d)
        tau_fog = np.exp(-gamma_fog * self.d)

        return tau_rain, tau_fog

    def cal_deg(self):
        deg_rain_streak = self.rain_streak * self.tau_fog

        tau = self.tau_fog * self.tau_rain
        rain_fog = self.img * tau + self.a * (1 - tau)

        self.deg = rain_fog + deg_rain_streak

    def synthesize(self):
        self.cal_deg()
        self.cal_metric()

    def cal_metric(self):

        psnr_value, ssim_value = calculate_psnr_ssim(self.img, self.deg)
        print(f"PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}")

    def visualize(self):
        fig, axes = plt.subplots(3, 2, figsize=(20, 10))

        # original background
        axes[0, 0].imshow(to_visualize(self.img))
        axes[0, 0].set_title('Background Signal')
        axes[0, 0].axis('off')

        # degenerate rain streak
        deg_rain_streak = self.rain_streak * self.tau_fog
        axes[0, 1].imshow(to_visualize(deg_rain_streak))
        axes[0, 1].set_title('Rain Streak')
        axes[0, 1].axis('off')

        # add rain streak
        rain_streak = self.img + deg_rain_streak
        rain_streak = np.clip(rain_streak, 0, 1)
        axes[1, 0].imshow(to_visualize(rain_streak))
        axes[1, 0].set_title('With Rain Streak')
        axes[1, 0].axis('off')

        # Fog
        fog = self.img * self.tau_fog + self.a * (1 - self.tau_fog)
        fog = np.clip(fog, 0, 1)
        axes[1, 1].imshow(to_visualize(fog))
        axes[1, 1].set_title('With Fog')
        axes[1, 1].axis('off')

        # Rain & Fog
        tau = self.tau_fog * self.tau_rain
        rain_fog = self.img * tau + self.a * (1 - tau)
        rain_fog = np.clip(rain_fog, 0, 1)
        axes[2, 0].imshow(to_visualize(rain_fog))
        axes[2, 0].set_title('With Fog and Rain')
        axes[2, 0].axis('off')

        # Final simulated image
        I = rain_fog + deg_rain_streak
        I = np.clip(I, 0, 1)
        axes[2, 1].imshow(to_visualize(I))
        axes[2, 1].set_title('Final Simulated Image ($I(x, y)$)')
        axes[2, 1].axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    model = RainModel(
        img_path='demo/BSD300/2092.jpg',
        r0=0.248,
        level=1,
        a=0.8
    )

    model.visualize()
    model.synthesize()
