# models/placeholder_model.py
import numpy as np

def predict_placeholder(img_arr, num_classes=2):
    """
    Return deterministic pseudo-probabilities for testing UI.
    The output shape is (num_classes,) and sums to 1.
    """
    s = int(img_arr.sum()) % 100
    probs = np.ones(num_classes) * 1e-6
    idx = s % num_classes
    probs[idx] = 0.9
    probs = probs / probs.sum()
    return probs
