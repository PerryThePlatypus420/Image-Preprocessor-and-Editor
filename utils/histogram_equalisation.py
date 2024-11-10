import numpy as np
import cv2
from PIL import Image
from utils.functions import pil_to_opencv


def histogram_equalization(img):
    # Convert PIL image to OpenCV format
    img_array = pil_to_opencv(img)

    # Convert the image to grayscale if it's in color
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # Calculate the histogram using NumPy's bincount
    H = np.bincount(img_array.flatten(), minlength=256)

    # Calculate the cumulative distribution function (CDF)
    CDF = np.cumsum(H)

    # Normalize the CDF
    CDF_normalized = CDF / float(CDF.max())

    # Perform histogram equalization
    equalized_img_array = np.interp(img_array.flatten(), range(
        256), CDF_normalized * 255).astype(np.uint8)
    equalized_img_array = equalized_img_array.reshape(img_array.shape)

    # Create a new PIL Image from the equalized array
    equalized_img = Image.fromarray(equalized_img_array)

    return equalized_img
