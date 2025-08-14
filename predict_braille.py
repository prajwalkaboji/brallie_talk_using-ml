import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Configuration
MODEL_PATH = "E:/bml/braille_character_model.h5"
IMG_SIZE = (28, 28)
label_map = {chr(i+65): i for i in range(26)}
reverse_map = {v: k for k, v in label_map.items()}

def predict_braille_image(image_path):
    # Load trained model
    model = load_model(MODEL_PATH)

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

# Example usage
if __name__ == "__main__":
    image_path = input("Enter Braille image path: ").strip()
    predict_braille_image(image_path)
