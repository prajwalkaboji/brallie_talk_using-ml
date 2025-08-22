import os
import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

IMG_SIZE = 64
DATASET_PATH = r"D:\bml\realtime_augmented"
MODEL_NAME = r"D:\bml\vgg16_braille_numbers.keras"

# -------------------------------
# 1. Load Dataset
# -------------------------------
def load_dataset(path=DATASET_PATH, img_size=64):
    data, labels = [], []
    # ✅ Only keep 0–9 folders
    classes = [str(i) for i in range(1,5)]  

    for idx, c in enumerate(classes):
        class_dir = os.path.join(path, c)
        if not os.path.isdir(class_dir):
            continue
        for img_name in os.listdir(class_dir):
            if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_size, img_size))
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # VGG expects 3 channels
            img = img.astype("float32") / 255.0
            data.append(img)
            labels.append(idx)

    data = np.array(data)
    labels = to_categorical(np.array(labels), num_classes=len(classes))
    return data, labels, classes

# -------------------------------
# 2. Build VGG16 Model
# -------------------------------
def build_vgg16(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=10):
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)

    # Freeze convolutional base
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom head
    x = Flatten()(base_model.output)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# -------------------------------
# 3. Training
# -------------------------------
def train_model():
    print("[📦] Loading dataset...")
    X, y, classes = load_dataset(DATASET_PATH, IMG_SIZE)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[🧠] Building VGG16...")
    num_classes = y.shape[1]
    model = build_vgg16((IMG_SIZE, IMG_SIZE, 3), num_classes)

    print("[🚀] Training...")
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

    model.save(MODEL_NAME)
    print(f"[💾] Model saved as {MODEL_NAME}")

    loss, acc = model.evaluate(X_test, y_test)
    print(f"[🎯] Test Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    train_model()
