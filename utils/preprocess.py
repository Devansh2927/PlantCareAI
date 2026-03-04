# utils/preprocess.py
from tensorflow.keras.preprocessing import image
import numpy as np

def preprocess_image(path, target_size=(224,224)):
    """
    Load image from path, resize to target_size, scale to [0,1],
    and return a batch of shape (1, H, W, 3).
    """
    img = image.load_img(path, target_size=target_size)
    arr = image.img_to_array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, 0)
    return arr
