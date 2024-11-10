import numpy as np
from PIL import Image

def log_transform(img, c):
    img_array = np.array(img, dtype=np.float32)  # Use float for precision in log calculation
    log_transformed_image = c * np.log1p(img_array)  # np.log1p is more precise for small values
    log_transformed_image = np.clip(log_transformed_image, 0, 255).astype(np.uint8)  # Clip values and convert to uint8
    return Image.fromarray(log_transformed_image)

def power_law_transform(img, g):
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize and ensure float32 for precision
    gamma_corrected_image = np.power(img_array, g) * 255.0
    gamma_corrected_image = np.clip(gamma_corrected_image, 0, 255).astype(np.uint8)  # Clip values and convert to uint8
    return Image.fromarray(gamma_corrected_image)

def negative_of_image(img):
    if img.mode == 'RGBA':
        # Separate the alpha channel
        r, g, b, a = img.split()
        rgb_image = Image.merge('RGB', (r, g, b))
        
        # Invert the RGB channels
        inverted_rgb_array = 255 - np.array(rgb_image)
        inverted_rgb_image = Image.fromarray(inverted_rgb_array)

        # Merge back the alpha channel
        result = Image.merge('RGBA', (*inverted_rgb_image.split(), a))
    else:
        img_array = np.array(img)
        negative_image_array = 255 - img_array
        result = Image.fromarray(negative_image_array)

    return result
