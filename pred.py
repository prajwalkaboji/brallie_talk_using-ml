import os
import cv2
import sys
import numpy as np
from tensorflow.keras.models import load_model

# =========================
# CONFIGURATION
# =========================
MODEL_PATH = r"D:\bml\vgg16_braille_numbers.keras"
IMG_SIZE = (64, 64)

# Label mapping for 1–4
label_map = ["1", "2", "3", "4"]

# =========================
# LOAD MODEL
# =========================
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model file not found: {MODEL_PATH}")
    sys.exit(1)

print(f"✅ Loading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)

# =========================
# PREDICTION FUNCTION
# =========================
def predict_braille_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return None

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # match training input
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    predicted_label = label_map[predicted_class]

    print(f"✅ Predicted Braille Number: {predicted_label} ({confidence*100:.2f}%)")
    return predicted_label

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        image_path = input("Enter Braille image path: ").strip()
    else:
        image_path = sys.argv[1]

    predict_braille_image(image_path)
