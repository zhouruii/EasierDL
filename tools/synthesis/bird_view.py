import matplotlib.pyplot as plt
import numpy as np

from tools.synthesis.rain_3d import generate_3d_rain


def generate_perspective_projection(proj='far', points=None, height=None, width=None, f=500, k=2.0):
    """
    Generate a bird's-eye view with perspective projection and intensity adjustment.

    Args:
        proj: Select which plane to project, 'far' or 'near'.
        points: A 3D numpy array of shape (N, 3) representing the point cloud.
                Each point has (x, y, z) coordinates.
        height: Height of the resulting bird's eye view image (in pixels).
        width: Width of the resulting bird's eye view image (in pixels).
        f: Focal length controlling the perspective effect.
        k: Non-linear exponent for intensity adjustment (higher values emphasize upper layers).

    Returns:
        A 2D numpy array representing the perspective-projected bird's eye view.
    """
    # Extract x, y, z coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    z_max = z.max()

    # Center the points
    x_center, y_center = width // 2, height // 2  # x.mean(), y.mean()
    x = x - x_center
    y = y - y_center

    # Initialize output image
    image = np.zeros((height, width), dtype=np.float32)

    # Image center
    cx, cy = width // 2, height // 2

    # Avoid division by zero by setting a minimum z value
    z[z < 1] = 1

    # z_eff = np.clip(z - 300, 1, None)

    # Perspective projection
    if proj == 'far':
        u = (f * x / z + cx).astype(int)  # Projected x coordinate
        v = (f * y / z + cy).astype(int)  # Projected y coordinate
    elif proj == 'near':
        u = (z * x / f + cx).astype(int)  # Projected x coordinate
        v = (z * y / f + cy).astype(int)  # Projected y coordinate
    else:
        raise NotImplementedError(f'Projection: {proj}, not supported yet')

    # Compute intensity using non-linear mapping
    intensity = (255 * (z / z_max) ** k).astype(np.float32)

    # Ensure points fall within image bounds and perform depth sorting
    for i in range(len(points)):
        if 0 <= u[i] < width and 0 <= v[i] < height:
            # Use the brighter (higher z) point at each pixel
            if intensity[i] > image[v[i], u[i]]:
                image[v[i], u[i]] = intensity[i]

    # Normalize the image to 0-255
    image = np.clip(image, 0, 255).astype(np.uint8)

    return image


if __name__ == '__main__':
    # Test with synthetic data
    N = 100000  # Number of points
    rain_3d = generate_3d_rain(512, 512, 512, num_drops=3000, streak_length=30, wind_angle=60, wind_strength=0.2)
    points = np.vstack(rain_3d).T  # Generate random points in the range (512, 512, 100)

    # Image dimensions
    height, width = 512, 512

    # Generate bird's eye view
    birdseye_image = generate_perspective_projection(proj='far', points=points, height=height, width=width, f=500)

    # Visualize the result
    plt.figure(figsize=(8, 8))
    plt.imshow(birdseye_image, cmap='gray')
    plt.title("Bird's Eye View with High-Altitude Emphasis")
    plt.axis("off")
    plt.show()
