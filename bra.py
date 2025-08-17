import os
import cv2
import numpy as np
import random
import argparse
import json
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

# ====== CONFIG ======
IMG_SIZE = 64
DATASET_DIR = r"C:\Users\Srushti\dataset_1to10"
MODEL_H5 = "braille_model.h5"
CLASS_MAP_FILE = "class_map.json"

# ====== BRAILLE DRAWING FUNCTIONS ======
def draw_braille_number(num, img_size=64):
    img = np.ones((img_size, img_size), dtype=np.uint8) * 255
    dot_radius = img_size // 10
    spacing = img_size // 4

    patterns = {
        1: [1],
        2: [1, 2],
        3: [1, 4],
        4: [1, 4, 5],
        5: [1, 5],
        6: [1, 2, 4],
        7: [1, 2, 4, 5],
        8: [1, 2, 5],
        9: [2, 4],
        10: [2, 4, 5]
    }

    coords = {
        1: (spacing, spacing),
        2: (spacing, spacing * 2),
        3: (spacing, spacing * 3),
        4: (spacing * 2, spacing),
        5: (spacing * 2, spacing * 2),
        6: (spacing * 2, spacing * 3)
    }

    for dot in patterns[num]:
        cv2.circle(img, coords[dot], dot_radius, (0,), -1)
    return img

def make_dataset():
    os.makedirs(DATASET_DIR, exist_ok=True)
    print(f"📦 Creating dataset under: {DATASET_DIR}")
    for num in range(1, 11):
        cls_dir = os.path.join(DATASET_DIR, "train", str(num))
        os.makedirs(cls_dir, exist_ok=True)
        print(f"  → Generating class '{num}'")
        for i in range(200):
            img = draw_braille_number(num, IMG_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # augmentation
            M = cv2.getRotationMatrix2D((IMG_SIZE//2, IMG_SIZE//2), random.uniform(-15,15), 1)
            img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE), borderValue=(255,255,255))
            img_path = os.path.join(cls_dir, f"{num}_{i}.png")
            cv2.imwrite(img_path, img)

# ====== MODEL ======
def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=10):
    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=input_shape),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def train_model():
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_flow = datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "train"),
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        class_mode="categorical",
        subset="training",
        batch_size=32,
        shuffle=True
    )
    val_flow = datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "train"),
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        class_mode="categorical",
        subset="validation",
        batch_size=32,
        shuffle=False
    )

    # Save class map
    INV_CLASS_MAP = {v: k for k, v in train_flow.class_indices.items()}
    with open(CLASS_MAP_FILE, "w") as f:
        json.dump(INV_CLASS_MAP, f)
    print("✅ Saved class map ->", CLASS_MAP_FILE)

    model = build_model()
    model.summary()

    checkpoint = ModelCheckpoint(MODEL_H5, monitor="val_accuracy", save_best_only=True, verbose=1)
    model.fit(train_flow, validation_data=val_flow, epochs=10, callbacks=[checkpoint])

# ====== REALTIME ======
def preprocess_for_model(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    return np.expand_dims(img, axis=0)

def realtime():
    model = load_model(MODEL_H5)
    with open(CLASS_MAP_FILE, "r") as f:
        IDX2LABEL = json.load(f)

    cap = cv2.VideoCapture(0)
    prob_buffer = []

    save_dir = os.path.join(DATASET_DIR, "train")
    os.makedirs(save_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        roi = frame[50:300, 50:300]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

        inp = preprocess_for_model(gray)
        prob = model.predict(inp, verbose=0)[0]
        prob_buffer.append(prob)
        if len(prob_buffer) > 5:
            prob_buffer.pop(0)
        avg_prob = np.mean(prob_buffer, axis=0)
        idx = int(np.argmax(avg_prob))
        conf = float(avg_prob[idx]) * 100.0
        pred_label = IDX2LABEL[str(idx)]

        cv2.rectangle(frame, (50,50), (300,300), (0,255,0), 2)
        cv2.putText(frame, f"{pred_label} ({conf:.1f}%)", (50,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Realtime Braille", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):  # save ROI
            class_folder = os.path.join(save_dir, pred_label)
            os.makedirs(class_folder, exist_ok=True)
            cv2.imwrite(os.path.join(class_folder, f"{pred_label}_{len(os.listdir(class_folder))}.png"), gray)
            print(f"💾 Saved sample to {class_folder}")

    cap.release()
    cv2.destroyAllWindows()

# ====== MAIN ======
def main():
    parser = argparse.ArgumentParser(description="Braille numbers (1–10): dataset, training, realtime.")
    parser.add_argument("--make_data", action="store_true", help="Generate synthetic dataset.")
    parser.add_argument("--train", action="store_true", help="Train the CNN.")
    parser.add_argument("--realtime", action="store_true", help="Run realtime webcam prediction.")
    parser.add_argument("--all", action="store_true", help="Generate dataset, train, then run realtime.")
    args = parser.parse_args()

    if args.make_data:
        make_dataset()
    if args.train:
        train_model()
    if args.realtime:
        realtime()
    if args.all:
        make_dataset()
        train_model()
        realtime()

if __name__ == "__main__":
    main()
