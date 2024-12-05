import cv2
import numpy as np
from noise import pnoise2


def generate_rain(height, width, num_drops=1000, drop_length=20, angle=10, intensity=0.5):
    """
    Generate realistic rain streaks.
    Args:
        height: Image height.
        width: Image width.
        num_drops: Number of rain drops.
        drop_length: Length of each rain drop (motion blur).
        angle: Angle of the rain streaks (in degrees).
        intensity: Intensity of the rain [0, 1].
    Returns:
        Rain streak image.
    """
    # Create a black image
    rain = np.zeros((height, width), dtype=np.uint8)

    # Generate random positions for rain drops
    for _ in range(num_drops):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        rain[y, x] = 255

    # Define a kernel for motion blur
    size = drop_length
    kernel = np.zeros((size, size))
    direction = np.deg2rad(angle)
    for i in range(size):
        x = int(i * np.sin(direction))
        y = int(i * np.cos(direction))
        kernel[y, x] = 1
    kernel /= kernel.sum()

    # Apply motion blur to the rain streaks
    rain_blurred = cv2.filter2D(rain, -1, kernel)

    # Adjust intensity
    rain_streaks = cv2.normalize(rain_blurred, None, 0, 255 * intensity, cv2.NORM_MINMAX)
    return rain_streaks.astype(np.uint8)


def generate_perlin_rain(height, width, scale=50, intensity=0.5, drop_length=20):
    """
    Generate rain streaks using Perlin noise.
    Args:
        height: Image height.
        width: Image width.
        scale: Scale factor for the Perlin noise.
        intensity: Intensity of the rain [0, 1].
        drop_length: Length of the rain streaks.
    Returns:
        Rain streak image.
    """
    # Create Perlin noise
    rain_noise = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            rain_noise[y, x] = pnoise2(x / scale, y / scale, octaves=4, repeatx=width, repeaty=height)

    # Normalize noise to [0, 255]
    rain_noise = (rain_noise - rain_noise.min()) / (rain_noise.max() - rain_noise.min()) * 255
    rain_noise = cv2.normalize(rain_noise, None, 0, 255 * intensity, cv2.NORM_MINMAX).astype(np.uint8)

    # Add motion blur
    kernel = np.ones((1, drop_length)) / drop_length
    rain_blurred = cv2.filter2D(rain_noise, -1, kernel)

    return rain_blurred


def generate_rain_in_high_altitude(height, width, num_drops=1000, drop_length=20, angle=10, intensity=0.5):
    """
    Generate realistic rain with a perspective effect, simulating high-altitude views.
    Args:
        height: Image height.
        width: Image width.
        num_drops: Number of rain drops.
        drop_length: Length of each rain drop (motion blur).
        angle: Angle of the rain streaks (in degrees).
        intensity: Intensity of the rain [0, 1].
    Returns:
        High altitude rain streak image.
    """
    # Create a black image
    rain = np.zeros((height, width), dtype=np.uint8)

    # Generate random positions for rain drops with perspective effect
    for _ in range(num_drops):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        rain[y, x] = 255

    # Define a kernel for motion blur (with perspective effect)
    size = drop_length
    kernel = np.zeros((size, size))
    direction = np.deg2rad(angle)
    for i in range(size):
        x = int(i * np.sin(direction))
        y = int(i * np.cos(direction))
        kernel[y, x] = 1
    kernel /= kernel.sum()

    # Apply motion blur to the rain streaks
    rain_blurred = cv2.filter2D(rain, -1, kernel)

    rain_blurred = cv2.normalize(rain_blurred, None, 0, 255 * intensity, cv2.NORM_MINMAX)

    # Apply perspective distortion for high-altitude view (simulating far-away rain)
    rain_blurred_perspective = cv2.resize(rain_blurred, (width // 2, height // 2))
    rain_blurred_perspective = cv2.resize(rain_blurred_perspective, (width, height), interpolation=cv2.INTER_LINEAR)

    return rain_blurred_perspective.astype(np.uint8)


def generate_perspective_rain_effect(height, width, num_drops=1000, drop_length=20, intensity=0.5,
                                     perspective_factor=0.8):
    """
    Simulate a rain effect with perspective and varying intensity.
    Args:
        height: Image height.
        width: Image width.
        num_drops: Number of rain drops.
        drop_length: Length of each rain drop.
        intensity: Intensity of the rain.
        perspective_factor: A factor to simulate perspective (near-far scaling).
    """
    rain = np.zeros((height, width), dtype=np.uint8)

    for _ in range(num_drops):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)

        # Simulating the intensity of the rain based on distance (perspective effect)
        rain_strength = (np.random.rand() * intensity) * (1 + (y / height) * (1 - perspective_factor))
        rain[y, x] = int(255 * rain_strength)

    # Apply motion blur
    kernel = np.ones((1, drop_length)) / drop_length
    rain_blurred = cv2.filter2D(rain, -1, kernel)

    return rain_blurred


# Example usage
if __name__ == "__main__":
    height, width = 512, 512

    rain = generate_rain(height, width, num_drops=2500, drop_length=10, angle=20, intensity=0.5)
    cv2.imwrite("./demo/Rain_Streaks.jpg", rain)

    # rain_perlin = generate_perlin_rain(height, width, scale=100, intensity=0.7, drop_length=20)
    # cv2.imwrite("./demo/Rain_Streaks_Perlin.jpg", rain_perlin)

    # rain = generate_rain_in_high_altitude(height, width, num_drops=1500, drop_length=5, angle=15, intensity=5.0)
    # cv2.imwrite("./demo/Rain_Streaks_in_high_altitude.jpg", rain)

    # rain = generate_perspective_rain_effect(height, width, num_drops=1500, drop_length=20, intensity=0.9,
    #                                         perspective_factor=0.8)
    # cv2.imwrite("./demo/Rain_Streaks_rain_effect.jpg", rain)
