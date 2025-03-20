import random
from collections import OrderedDict

import cv2
import numpy as np

from gen_perlin import generate_perlin_noise
from gen_streak import generate_bird_view_streak
from gif import guided_filter
from load import load_pickle, load_hsi
from util import calculate_psnr_ssim, check_dtype, visualize_tool, normalize

from config import DV, RAIN, LEVEL

BANDS = [36, 19, 8]

RAIN_STREAK = {
    1: dict(num_drops=random.randint(2000, 2500), streak_length=random.randint(20, 25),
            wind_angle=random.randint(-180, 180), wind_strength=random.uniform(0, 0.05)),  # 小雨
    2: dict(num_drops=random.randint(3000, 3500), streak_length=random.randint(30, 35),
            wind_angle=random.randint(-180, 180), wind_strength=random.uniform(0.05, 0.1)),  # 中雨
    3: dict(num_drops=random.randint(3500, 4000), streak_length=random.randint(40, 45),
            wind_angle=random.randint(-180, 180), wind_strength=random.uniform(0.1, 0.2)),  # 大雨
    4: dict(num_drops=random.randint(5000, 5500), streak_length=random.randint(50, 55),
            wind_angle=random.randint(-180, 180), wind_strength=random.uniform(0.2, 0.3)),  # 暴雨
}


class RainModelForAVIRIS:
    def __init__(self,
                 hsi_path=None,
                 bands_path=None,
                 r0=None,
                 level=None,
                 d=0.3,
                 a=1.0,
                 gif=False,
                 use_perlin=True,
                 alpha=1.0
                 ):
        self.hsi = load_hsi(hsi_path)
        # self.hsi = normalize(self.hsi)
        self.bands_path = bands_path
        self.r0 = random.uniform(0, 0.66) if not r0 else r0
        self.level = random.uniform(1, 3) if not level else level
        self.d = np.full(self.hsi.shape, d)
        self.a = np.full(self.hsi.shape, a)
        self.gif = gif
        self.use_perlin = use_perlin
        self.alpha = alpha

        self.height, self.width, self.channel = self.hsi.shape
        self.depth = (self.height + self.width) // 2
        self.dv = DV.get(self.level)
        self.rain_speed = RAIN.get(self.level)
        self.label = LEVEL.get(self.level)
        self.lambdas = load_pickle(self.bands_path)['bands']

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
        rain_streak = generate_bird_view_streak(height=self.height,
                                                width=self.width,
                                                depth=self.depth,
                                                f=self.depth,
                                                **RAIN_STREAK.get(self.level))
        rain_streak = check_dtype(rain_streak)
        if len(rain_streak.shape) < 3:
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
        psnr_value, ssim_value = calculate_psnr_ssim(self.hsi, self.deg_img)
        print(f"PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}")

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
                       save_path=f'result/AVIRIS/{self.label}.jpg',
                       RGB=RGB)


if __name__ == '__main__':
    # PSNR: 10 15 20 25
    # alpha: 0.7 0.7 0.8 0.9
    seed = 42
    random.seed(seed)

    model = RainModelForAVIRIS(
        hsi_path='/home/disk1/ZR/PythonProjects/uchiha/tools/hyperspectral/data/f130411t01p00r11rdn_e_11.npy',
        bands_path='/home/disk1/ZR/PythonProjects/uchiha/tools/hyperspectral/meta.pkl',
        r0=0.155,  # 0.248
        level=1,
        a=1,
        d=0.4,
        gif=True,
        alpha=0.7
    )

    model.synthesize()
    model.visualize(RGB=True)
