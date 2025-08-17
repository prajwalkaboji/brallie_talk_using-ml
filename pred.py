import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# =========================
# CONFIGURATION
# =========================
MODEL_PATH = r"D:\bml\braille_character_minivgg.h5"  # Matches your training save path
IMG_SIZE = (28, 28)
label_map = {chr(i + 65): i for i in range(26)}
reverse_map = {v: k for k, v in label_map.items()}

# =========================
# LOAD MODEL ONCE
# =========================
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model file not found: {MODEL_PATH}")
    exit()

print(f"✅ Loading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)

# =========================
# PREDICTION FUNCTION
# =========================
def predict_braille_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return None

    # Read and preprocess image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)  # shape: (28, 28, 1)
    img = np.expand_dims(img, axis=0)   # shape: (1, 28, 28, 1)

    # Predict
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    predicted_label = reverse_map[predicted_class]

    print(f"Predicted Braille Character: {predicted_label}")
    return predicted_label

# =========================
# MAIN SCRIPT
# =========================
if __name__ == "__main__":
    image_path = input("Enter Braille image path: ").strip()
    predict_braille_image(image_path)
