import cv2
import numpy as np
import matplotlib.pyplot as plt
from albumentations import Compose, RandomRain, RandomFog, RandomShadow

def apply_rain_fog_shadow_effects(image_path):
    """
    Apply rain, fog, and shadow effects to an input image using Albumentations.
    Args:
        image_path: Path to the input image.
    Returns:
        Augmented image with rain, fog, and shadow effects.
    """
    # Load the input image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define augmentation pipeline
    augmentation = Compose([
        # Add rain
        RandomRain(
            slant_lower=-20,  # Rain angle: more slanted
            slant_upper=20,
            drop_length=15,  # Moderate rain drop length
            drop_width=1,  # Thin drops
            drop_color=(200, 200, 200),  # Light gray
            blur_value=3,  # Slight motion blur
            brightness_coefficient=0.9,  # Reduce overall brightness slightly
            rain_type='heavy',  # Type: heavy, drizzle, or torrential
        ),
        # Add fog
        RandomFog(
            fog_coef_lower=0.1,  # Lower fog density
            fog_coef_upper=0.3,  # Upper fog density
            alpha_coef=0.1  # Control fog transparency
        ),
        # Add shadow
        RandomShadow(
            shadow_roi=(0, 0.5, 1, 1),  # Shadow area (bottom half of the image)
            num_shadows_lower=1,  # Number of shadows
            num_shadows_upper=2,
            shadow_dimension=5,  # Shadow blur
        )
    ])

    # Apply the augmentations
    augmented = augmentation(image=image)["image"]

    return augmented

# Visualize the result
def visualize_augmented_effect(image_path):
    """
    Visualize the augmented rain, fog, and shadow effects.
    Args:
        image_path: Path to the input image.
    """
    # Apply effects
    augmented_image = apply_rain_fog_shadow_effects(image_path)

    # Show original and augmented image side by side
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(augmented_image)
    plt.title("Rain, Fog, and Shadow Effects")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# Test the visualization
image_path = 'demo/BSD300/2092.jpg' # Replace with your image path
visualize_augmented_effect(image_path)
