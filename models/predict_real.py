# models/predict_real.py
import os
import numpy as np
from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.h5")

# module-level cache for the loaded model
_model = None

def _load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        _model = keras.models.load_model(MODEL_PATH)
    return _model

def predict_real(img_arr):
    """
    Predict on a preprocessed image array.
    Returns: (pred_probs, gradcam_filename_or_None)
    - pred_probs: numpy array shape (1, num_classes)
    - gradcam_filename_or_None: string or None
    """
    model = _load_model()
    preds = model.predict(img_arr)
    # If you don't have Grad-CAM implemented yet, return None for filename
    gradcam_filename = None
    return preds, gradcam_filename

