import cv2
from utils.functions import pil_to_opencv, opencv_to_pil

def apply_otsu_thresholding(img):
    # Convert PIL image to OpenCV format (assumed to be in color)
    cv_img = pil_to_opencv(img)
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    # Apply Otsu's thresholding
    _, thresholded_image = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Convert the thresholded OpenCV image back to PIL format
    thresholded_pil_image = opencv_to_pil(thresholded_image)
    
    return thresholded_pil_image
