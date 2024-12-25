import cv2
import numpy as np
from matplotlib import pyplot as plt

from tools.synthesis.bird_view import generate_perspective_projection
from tools.synthesis.gen_perlin import generate_perlin_noise
from tools.synthesis.rain_3d import generate_3d_rain
from tools.synthesis.util import normalize


def smooth_image(image, method="gaussian", kernel_size=5):
    """
    Apply smoothing to an image to reduce artifacts and sharp transitions.
    Args:
        image: Input image (grayscale).
        method: Smoothing method ("gaussian" or "bilateral").
        kernel_size: Kernel size for smoothing.
    Returns:
        Smoothed image.
    """
    if method == "gaussian":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif method == "bilateral":
        return cv2.bilateralFilter(image, kernel_size, 75, 75)
    else:
        raise ValueError("Invalid smoothing method specified!")


def generate_rain_streak(height, width, num_drops=1000, drop_length=20, angle=10, intensity=0.5, mask=None, noise=None):
    """
    Generate realistic rain streaks with Perlin noise.
    Args:
        height: Image height.
        width: Image width.
        num_drops: Number of rain drops.
        drop_length: Length of each rain drop (motion blur).
        angle: Angle of the rain streaks (in degrees).
        intensity: Intensity of the rain [0, 1].
        mask: Mask of the Perlin noise for rain density.
        noise: Perlin noise for rain density.
    Returns:
        Rain streak image.
    """

    # Threshold the noise map to create rain drop mask
    streak = np.zeros((height, width))

    # Generate random positions for rain drops
    if mask is not None and noise is None:
        drop_positions = np.column_stack(np.where(mask > 0))
        for _ in range(min(num_drops, drop_positions.shape[0])):
            y, x = drop_positions[np.random.choice(len(drop_positions))]
            streak[y, x] = 255
    else:
        for _ in range(num_drops):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            streak[y, x] = 255

    # Define a kernel for motion blur
    size = drop_length
    kernel = np.zeros((size, size), dtype=np.float32)
    direction = np.deg2rad(angle)
    for i in range(size):
        x = int(i * np.sin(direction))
        y = int(i * np.cos(direction))
        if 0 <= y < size and 0 <= x < size:
            kernel[y, x] = 1
    kernel /= kernel.sum()

    # Apply motion blur to the rain streaks
    rain_blurred = cv2.filter2D(streak, -1, kernel)

    # Adjust intensity
    rain_streaks = normalize(rain_blurred, 0, 255 * intensity).astype(np.uint8)

    if noise is not None:
        noise = normalize(noise.astype(np.float32), 0.3, 1)
        rain_streaks = (rain_streaks * noise).astype(np.uint8)

    return smooth_image(rain_streaks)


def generate_top_streak(height, width, num_streaks=2000, center_fraction=0.05, max_streak_length=30):
    """
    Generate a top-view rain simulation in grayscale.
    Args:
        height: Image height.
        width: Image width.
        num_streaks: Total number of rain streaks.
        center_fraction: Fraction of the image representing the circular center region with raindrops.
        max_streak_length: Maximum length of the rain streaks at the edges.
    Returns:
        A 2D grayscale rain image with discrete radial streaks and gradient effects.
    """
    # Create a blank grayscale image
    rain_image = np.zeros((height, width), dtype=np.float32)

    # Center coordinates
    center_x, center_y = width // 2, height // 2
    max_distance = np.sqrt(center_x ** 2 + center_y ** 2)

    # ** Step 1: Generate sparse circular raindrops in the center **
    center_radius = int(center_fraction * min(height, width) / 2)
    num_center_drops = int(num_streaks * center_fraction)
    for _ in range(num_center_drops):
        x_center = np.random.randint(center_x - center_radius, center_x + center_radius)
        y_center = np.random.randint(center_y - center_radius, center_y + center_radius)
        if 0 <= x_center < width and 0 <= y_center < height:
            radius = np.random.randint(1, 3)  # Small radius
            intensity = np.random.randint(150, 200)  # Light gray raindrops
            # Add irregularity to the circles for a natural look
            irregularity = np.random.uniform(0.8, 1.2)
            cv2.circle(rain_image, (x_center, y_center), int(radius * irregularity), intensity, -1)

    # ** Step 2: Generate radial rain streaks **
    for _ in range(num_streaks):
        # Randomly sample a starting position for the streak
        distance = np.random.uniform(0, max_distance)
        angle = np.random.uniform(0, 2 * np.pi)

        x_start = int(center_x + distance * np.cos(angle))
        y_start = int(center_y + distance * np.sin(angle))

        # Ensure the starting point is within the image bounds
        if not (0 <= x_start < width and 0 <= y_start < height):
            continue

        # Determine streak length and direction
        streak_length = int(max_streak_length * (distance / max_distance))
        streak_length = max(5, streak_length)  # Ensure minimum streak length
        # dx = int(streak_length * np.cos(angle))
        # dy = int(streak_length * np.sin(angle))

        # Draw streak with pixel-level gradient
        for i in range(streak_length):
            xi = int(x_start + i * np.cos(angle))
            yi = int(y_start + i * np.sin(angle))
            if 0 <= xi < width and 0 <= yi < height:
                intensity = 200 + int((55 * i) / streak_length)  # Gradual transition from gray (200) to white
                rain_image[yi, xi] = intensity

    # Normalize the image to 0-255
    # rain_image = cv2.normalize(rain_image, None, 0, 255, cv2.NORM_MINMAX)
    rain_image = normalize(rain_image, 0, 255)
    return rain_image.astype(np.uint8)


def generate_bird_view_streak(height, width, depth, num_drops=10000, streak_length=20, wind_angle=135,
                              wind_strength=10, noise=None):
    rain_3d = generate_3d_rain(height=height,
                               width=width,
                               depth=depth,
                               num_drops=num_drops,
                               streak_length=streak_length,
                               wind_angle=wind_angle,
                               wind_strength=wind_strength)
    points = np.vstack(rain_3d).T
    bird_view = generate_perspective_projection(points=points,
                                                height=height,
                                                width=width,
                                                f=depth)

    if noise is not None:
        noise = normalize(noise.astype(np.float32), 0.3, 1)
        bird_view = (bird_view * noise).astype(np.uint8)

    return smooth_image(bird_view)


def visualize(params):
    """
    Visualize different rain types with varying intensities and parameters.
    """

    fig, axes = plt.subplots(1, len(params), figsize=(20, 5))

    for i, param in enumerate(params):
        # rain_image = generate_rain_streak(
        #     **param
        # )
        rain_image = generate_bird_view_streak(
            **param
        )
        axes[i].imshow(rain_image, cmap="gray", vmin=0, vmax=255)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    perlin_noise = generate_perlin_noise(impl='noise', height=512, width=512, scales=[600])
    perlin_mask = (perlin_noise > 68).astype(np.uint8)

    # RAIN = [
    #     {"height": 512, "width": 512, "num_drops": 1000, "drop_length": 10, "intensity": 0.3, "angle": 10,
    #      "mask": perlin_mask, "noise": perlin_noise},  # Small
    #     {"height": 512, "width": 512, "num_drops": 1200, "drop_length": 20, "intensity": 0.5, "angle": 15,
    #      "mask": perlin_mask, "noise": perlin_noise},  # Medium
    #     {"height": 512, "width": 512, "num_drops": 2000, "drop_length": 30, "intensity": 0.7, "angle": 20,
    #      "mask": perlin_mask, "noise": perlin_noise},  # Heavy
    #     {"height": 512, "width": 512, "num_drops": 3000, "drop_length": 40, "intensity": 0.9, "angle": 25,
    #      "mask": perlin_mask, "noise": perlin_noise},  # Storm
    # ]

    RAIN = [
        {"height": 512, "width": 512, "depth": 512, "num_drops": 1000, "streak_length": 10, "wind_angle": 0,
         "wind_strength": 0, "noise": perlin_noise},  # Small
        {"height": 512, "width": 512, "depth": 512, "num_drops": 1200, "streak_length": 20, "wind_angle": 10,
         "wind_strength": 5, "noise": perlin_noise},  # Medium
        {"height": 512, "width": 512, "depth": 512, "num_drops": 2000, "streak_length": 30, "wind_angle": 20,
         "wind_strength": 10, "noise": perlin_noise},  # Heavy
        {"height": 512, "width": 512, "depth": 512, "num_drops": 3000, "streak_length": 40, "wind_angle": 30,
         "wind_strength": 15, "noise": perlin_noise},  # Storm
    ]

    visualize(RAIN)
