import numpy as np
import cv2
import matplotlib.pyplot as plt


def add_fog(image, beta_lambda=0.1, F0=0.8, fog_intensity=0.5, noise_factor=0.2):
    """
    Add fog effect to an image using the refined fog model.

    :param image: Input image (numpy array)
    :param beta_lambda: The fog strength parameter, related to the wavelength (typically constant for simplicity)
    :param F0: Fog background intensity (0-1)
    :param fog_intensity: Fog intensity (0-1), controls the strength of the fog
    :param noise_factor: Noise factor to create a random fog map (0-1)
    :return: Image with fog effect
    """
    # Convert the input image to float32 for calculation
    image_float = image.astype(np.float32) / 255.0

    # Generate a random noise map for distance (simulate pixel-to-camera distance)
    height, width = image.shape[:2]
    # d_map = np.random.rand(height, width) * 1.0  # Example: Use distance as [0, 1]
    d_map = np.ones((height, width)) * 1.0

    # Fog strength function F(x, y, λ) = 1 - exp(-β(λ) * d)
    fog_map = 1 - np.exp(-beta_lambda * d_map)

    # Normalize the fog map between 0 and 1 (for safety)
    fog_map = np.clip(fog_map, 0, 1)

    # Apply the fog model I_F(x, y, λ) = I(x, y, λ) * (1 - F(x, y, λ)) + F0 * F(x, y, λ)
    fogged_image = image_float * (1 - fog_map[..., None]) + F0 * fog_map[..., None]

    # Ensure the result is within the valid range [0, 1]
    fogged_image = np.clip(fogged_image, 0, 1)

    # Convert the image back to uint8
    return (fogged_image * 255).astype(np.uint8)


# Load an example image
image = cv2.imread(r"D:\Uchiha\wallpaper\bay_10-wallpaper-3840x2160.jpg")

# Add fog effect to the image
foggy_image = add_fog(image, beta_lambda=0.5, F0=0.8, fog_intensity=0.5, noise_factor=0.2)

# Display the original and foggy images
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Foggy Image")
plt.imshow(cv2.cvtColor(foggy_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
