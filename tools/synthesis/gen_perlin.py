import numpy as np
from matplotlib import pyplot as plt
from noise import snoise2
from perlin_noise import PerlinNoise

from tools.synthesis.util import normalize


def generate_perlin_noise(impl='noise', height=512, width=512, scales=None, alpha=None):
    """
    Generate perlin noise.
    Args:
        impl: implement library. 'noise' or 'perlin_noise'
        height: Height of the noise map.
        width: Width of the noise map.
        scales: The scale of the noise (larger values -> more detail).
        alpha: Power control contrast.
    Returns:
        Perlin noise array [0~255].
    """
    # Create Perlin noise
    if scales is None:
        scales = [100]

    noise = np.zeros((height, width), dtype=np.float32)

    if impl == 'noise':
        for scale in scales:
            for y in range(height):
                for x in range(width):
                    noise_value = snoise2(x / scale, y / scale, octaves=10, repeatx=width, repeaty=height)
                    # Normalize to range [0, 1]
                    noise_value = (noise_value + 1) / 2
                    if alpha is None:
                        noise[y, x] = noise_value
                    else:
                        # Apply exponential transformation
                        noise_value = noise_value ** alpha
                    noise[y, x] += noise_value
    elif impl == 'perlin_noise':
        perlin_noise = PerlinNoise(octaves=4, seed=10)
        for scale in scales:
            for y in range(height):
                for x in range(width):
                    noise[y, x] += perlin_noise([y / scale, x / scale])
    else:
        raise NotImplementedError(f'impl: {impl} not supported')

    # Normalize to [0, 1]
    noise = normalize(noise, 0, 255).astype(np.uint8)

    return noise


def get_perlin_mask(noise_map, threshold=68):
    drop_mask = (noise_map > threshold).astype(np.uint8)  # Adjust threshold for density
    return drop_mask


def visualize():
    """
    Visualize different rain effects with Perlin noise.
    """

    fig, axes = plt.subplots(2, 2, figsize=(10, 5))

    noise1 = dict(impl='noise', height=512, width=512, scales=[450])
    noise2 = dict(impl='noise', height=512, width=512, scales=[500, 600], alpha=0.1)

    perlin1 = generate_perlin_noise(**noise1)
    perlin2 = generate_perlin_noise(**noise2)

    mask1 = get_perlin_mask(perlin1)
    mask2 = get_perlin_mask(perlin2)

    axes[0][0].imshow(perlin1, cmap='gray')
    axes[0][0].set_title('noise1')
    axes[0][0].axis("off")

    axes[0][1].imshow(perlin2, cmap='gray')
    axes[0][1].set_title('noise2')
    axes[0][1].axis("off")

    axes[1][0].imshow(mask1, cmap='gray')
    axes[1][0].set_title('mask')
    axes[1][0].axis("off")

    axes[1][1].imshow(mask2, cmap='gray')
    axes[1][1].set_title('mask')
    axes[1][1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    visualize()
