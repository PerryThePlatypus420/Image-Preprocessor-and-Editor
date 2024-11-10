import cv2
from utils.functions import pil_to_opencv, opencv_to_pil

def add_watermark(img, watermark):
    img = pil_to_opencv(img)
    watermark = pil_to_opencv(watermark)

    h_img, w_img, _ = img.shape
    h_wm, w_wm, _ = watermark.shape

    # Define watermark scale: Make it small, e.g., 5% of the image width
    wm_scale = 0.05 * w_img / max(h_wm, w_wm)  
    wm_dim = (int(w_wm * wm_scale), int(h_wm * wm_scale))
    resized_wm = cv2.resize(watermark, wm_dim, interpolation=cv2.INTER_AREA)

    # Position watermark at the bottom right corner
    bottom_y = h_img - wm_dim[1]
    right_x = w_img - wm_dim[0]
    roi = img[bottom_y:h_img, right_x:w_img]

    # Blend watermark with the ROI
    if resized_wm.shape[2] == 4:
        # Use the alpha channel for blending if watermark image is transparent
        alpha_channel = resized_wm[:, :, 3] / 255.0
        for i in range(3):
            roi[:, :, i] = roi[:, :, i] * (1 - alpha_channel) + resized_wm[:, :, i] * alpha_channel
    else:
        roi = cv2.addWeighted(roi, 1, resized_wm, 0.3, 0)

    img[bottom_y:h_img, right_x:w_img] = roi

    return opencv_to_pil(img)
