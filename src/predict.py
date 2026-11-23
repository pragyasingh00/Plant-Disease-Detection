
import tensorflow as tf
import numpy as np

from pathlib import Path
from PIL import Image


# --------------------------------
# 1. Configuration
# --------------------------------

MODEL_PATH = Path("models/mobilenet_best.keras")

DATASET_DIR = Path("dataset/processed_900/train")

IMAGE_SIZE = (224, 224)


# --------------------------------
# 2. Load trained model
# --------------------------------

def load_prediction_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    return model


# --------------------------------
# 3. Get class names
# --------------------------------

def get_class_names():

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR}"
        )

    class_names = sorted([
        folder.name
        for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
    ])

    return class_names


# --------------------------------
# 4. Preprocess uploaded image
# --------------------------------

def preprocess_image(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize image
    image = image.resize(IMAGE_SIZE)

    # Convert image into NumPy array
    image_array = np.array(
        image,
        dtype=np.float32
    )

    # MobileNetV2 preprocessing
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# --------------------------------
# 5. Predict plant disease
# --------------------------------

def predict_image(image, model=None):

    if model is None:
        model = load_prediction_model()

    class_names = get_class_names()

    processed_image = preprocess_image(image)

    predictions = model.predict(
        processed_image,
        verbose=0
    )

    predicted_index = int(
        np.argmax(predictions[0])
    )

    predicted_class = class_names[predicted_index]

    confidence = float(
        predictions[0][predicted_index] * 100
    )

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_probabilities": predictions[0].tolist()
    }
