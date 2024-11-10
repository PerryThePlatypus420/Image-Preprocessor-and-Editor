from sklearn.cluster import KMeans
import numpy as np
from PIL import Image

def quantize_image(img, num_colors):
    # Convert the image to a NumPy array and flatten
    img_array = np.array(img)
    original_shape = img_array.shape  # Store the original shape
    img_array = img_array.reshape((-1, 3))  # Flatten the image

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_colors, n_init=10).fit(img_array)

    # Replace each pixel with its nearest representative color
    quantized_pixels = kmeans.cluster_centers_[kmeans.labels_].astype(int)
    
    # Ensure reshaping back to original image dimensions
    if len(original_shape) == 3:  # Check if the image is color (3D)
        quantized_img_array = quantized_pixels.reshape(original_shape)
    else:  # Grayscale image (2D)
        quantized_img_array = quantized_pixels.reshape(original_shape[0], original_shape[1])

    return Image.fromarray(quantized_img_array.astype(np.uint8))
