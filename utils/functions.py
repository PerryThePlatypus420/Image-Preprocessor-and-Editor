import base64
import cv2
import numpy as np
from PIL import Image

def pil_to_opencv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def opencv_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def process_image(img, operation):
    cv_img = pil_to_opencv(img)
    processed_img = operation(cv_img)
    return opencv_to_pil(processed_img)

def resize_image(img, w, h):
    return process_image(img, lambda cv_img: resizeImg(cv_img, w, h))


def resizeImg(img, new_width, new_height):

    # Get the dimensions of the original image
    height, width = img.shape[:2]

    # Create an empty array for the resized image
    resized_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # Calculate scaling factors for height and width
    scale_y = height / new_height
    scale_x = width / new_width

    # Iterate over each pixel in the resized image
    for y in range(new_height):
        for x in range(new_width):
            # Calculate the corresponding position in the original image
            original_y = y * scale_y
            original_x = x * scale_x

            # Find the four nearest pixels in the original image
            y1 = int(original_y)
            y2 = min(y1 + 1, height - 1)
            x1 = int(original_x)
            x2 = min(x1 + 1, width - 1)

            # Bilinear interpolation
            dy = original_y - y1
            dx = original_x - x1

            # Interpolate values for each channel
            interpolated_pixel = (
                (1 - dx) * (1 - dy) * img[y1, x1] +
                dx * (1 - dy) * img[y1, x2] +
                (1 - dx) * dy * img[y2, x1] +
                dx * dy * img[y2, x2]
            )

            # Set the pixel value in the resized image
            resized_image[y, x] = np.round(interpolated_pixel).astype(np.uint8)

    return resized_image
            

def rotate_cropped(img, angle):
    def rotate_cv(cv_img):
        center = tuple(np.array(cv_img.shape[1::-1]) / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(cv_img, rotation_matrix, cv_img.shape[1::-1], flags=cv2.INTER_LINEAR)

    return process_image(img, rotate_cv)

def rotate_not_cropped(img, angle):
    def rotate_cv(cv_img):
        center = tuple(np.array(cv_img.shape[1::-1]) / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        cosine, sine = np.abs(rotation_matrix[0, :2])
        new_width = int((cv_img.shape[0] * sine) + (cv_img.shape[1] * cosine))
        new_height = int((cv_img.shape[0] * cosine) + (cv_img.shape[1] * sine))

        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]

        return cv2.warpAffine(cv_img, rotation_matrix, (new_width, new_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    return process_image(img, rotate_cv)

def horizontal_flip(img):
    def h_flip(img):
        height, width = img.shape[:2]

        horizontal_flipped_image = np.zeros_like(img)

        for i in range(height):
            for j in range(width):
                horizontal_flipped_image[i, j] = img[i, width - 1 - j]
        
        return horizontal_flipped_image
    
    return process_image(img, h_flip)

def vertical_flip(img):
    def v_flip(img):
        height, width = img.shape[:2]

        vertical_flipped_image = np.zeros_like(img)

        for i in range(height):
            for j in range(width):
                vertical_flipped_image[i, j] = img[height - 1 - i, j]

        return vertical_flipped_image
    
    return process_image(img, v_flip)

def crop_image(img):
    def crop(cv_img):
        # Define the region of interest (ROI)
        x, y, w, h = 100, 50, 200, 150
        cropped_img = cv_img[y:y+h, x:x+w]
        height, width, _ = cv_img.shape
        return cv2.resize(cropped_img, (width, height))
    return process_image(img, crop)

def linearContrastStretch(img, Sma):
    def contrast_stretch(cv_img):
        Rmin = cv_img.min()
        Rmax = cv_img.max()
        Smi = 0
        S = (cv_img - Rmin) * ((Sma - Smi) / (Rmax - Rmin)) + Smi
        return S.astype(np.uint8)
    return process_image(img, contrast_stretch)

def image_brightness(img, bright):
    def adjust_brightness(cv_img):
        # Brightness adjustment
        return cv2.addWeighted(cv_img, 1.0, cv_img, 0, bright)
    return process_image(img, adjust_brightness)

def color_space_transform(img, target_color_space):
    def transform(cv_img):
        if target_color_space == 'RGB':
            return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        elif target_color_space == 'HSV':
            return cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        elif target_color_space == 'LAB':
            return cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
        else:
            raise ValueError(f"Unsupported target color space: {target_color_space}")
    return process_image(img, transform)

def convert_to_grayscale(img):
    
    def grayscale(img):
        # Ensure the image is in the correct format
        if len(img.shape) != 3 or img.shape[2] != 3:
            raise ValueError("Input must be a 3-channel BGR image")

        # Extract the BGR channels
        B = img[:,:,0]
        G = img[:,:,1]
        R = img[:,:,2]

        # Calculate luminance
        grayscale_image = 0.299 * R + 0.587 * G + 0.114 * B

        # Convert to uint8 (8-bit) data type
        grayscale_image = grayscale_image.astype(np.uint8)

        return grayscale_image
    
    return process_image(img, grayscale)

def gaussian_blur(img, n):
    return process_image(img, lambda cv_img: cv2.GaussianBlur(cv_img, (n, n), 0))

def median_blur(img, n):
    return process_image(img, lambda cv_img: cv2.medianBlur(cv_img, n))

import io

def get_image_bytes(img):
    """Converts a PIL image to bytes while preserving its original format"""
    buffered = io.BytesIO()

    # Determine the image format or default to PNG
    format = img.format if img.format else "PNG"

    img.save(buffered, format=format)
    return buffered.getvalue()