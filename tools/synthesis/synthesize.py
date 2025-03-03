from matplotlib import pyplot as plt

from tools.synthesis.rain_model import RainModel
from tools.synthesis.util import to_visualize, scale_streak, calculate_psnr_ssim

if __name__ == '__main__':
    model = RainModel(
        img_path='demo/5.jpg',
        r0=0.248,
        level=2,
        a=1,
        d=1,
        scale=300
    )

    img = model.img
    streak = model.rain_streak
    scaled_streak = scale_streak(streak)
    deg_img = img + scaled_streak

    psnr_value, ssim_value = calculate_psnr_ssim(img, deg_img)
    print(f'PSNR:{psnr_value}, SSIM:{ssim_value}')

    fig, axes = plt.subplots(1, 2, figsize=(10, 8))
    axes[0].imshow(to_visualize(img))
    axes[1].imshow(to_visualize(deg_img))

    plt.tight_layout()
    plt.show()
