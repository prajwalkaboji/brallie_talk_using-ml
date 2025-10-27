import os
import cv2
import numpy as np
import argparse
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# -------------------------------
# Settings
# -------------------------------
IMG_SIZE = 64
DATASET_PATH = r"E:\bml\cnn_resized"
MODEL_NAME = r"D:\bml\final.keras"


# -------------------------------
# 1. Load Dataset
# -------------------------------
def load_dataset(path=DATASET_PATH, img_size=64):
    data, labels = [], []
    classes = ["1", "2", "3", "4", "5"]  # ✅ Braille numbers to train

    for idx, c in enumerate(classes):
        class_dir = os.path.join(path, c)
        if not os.path.isdir(class_dir):
            print(f"[⚠️] Skipping missing folder: {class_dir}")
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
    print(f"[✅] Loaded {len(data)} images from {len(classes)} classes: {classes}")
    return data, labels, classes


# -------------------------------
# 2. Build VGG16 Model
# -------------------------------
def build_vgg16(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=5):  # ✅ 5 classes
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)

    # Freeze base layers
    for layer in base_model.layers:
        layer.trainable = False

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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("[🧠] Building VGG16...")
    model = build_vgg16((IMG_SIZE, IMG_SIZE, 3), num_classes=y.shape[1])

    print("[🚀] Training started...")
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

    model.save(MODEL_NAME)
    print(f"[💾] Model saved as {MODEL_NAME}")

    loss, acc = model.evaluate(X_test, y_test)
    print(f"[🎯] Test Accuracy: {acc*100:.2f}%")


# -------------------------------
# 4. Realtime Webcam Prediction
# -------------------------------
def realtime_prediction():
    if not os.path.exists(MODEL_NAME):
        print("[❌] No trained model found. Train first with --train")
        return

    model = load_model(MODEL_NAME)
    _, _, classes = load_dataset(DATASET_PATH, IMG_SIZE)
    cap = cv2.VideoCapture(0)

    print("[📷] Starting realtime recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        x, y, w, h = 100, 100, 200, 200
        roi = frame[y:y+h, x:x+w]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        gray = gray.astype("float32") / 255.0
        gray = np.expand_dims(gray, axis=0)

        pred = model.predict(gray, verbose=0)
        label_idx = np.argmax(pred)
        conf = np.max(pred)
        label = classes[label_idx]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Num: {label} ({conf*100:.1f}%)",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Realtime Braille Num Recognition (VGG16)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# -------------------------------
# 5. Predict Single Image
# -------------------------------
def predict_image(image_path):
    if not os.path.exists(MODEL_NAME):
        print("[❌] No trained model found. Train first with --train")
        return
    if not os.path.exists(image_path):
        print("[❌] Image path not found:", image_path)
        return

    model = load_model(MODEL_NAME)
    _, _, classes = load_dataset(DATASET_PATH, IMG_SIZE)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)
    label_idx = np.argmax(pred)
    conf = np.max(pred)
    label = classes[label_idx]

    print(f"[🖼️] Predicted Braille Number: {label}  |  Confidence: {conf*100:.2f}%")


# -------------------------------
# 6. Main
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--realtime", action="store_true", help="Run realtime webcam")
    parser.add_argument("--predict", type=str, help="Predict a single image (give image path)")
    args = parser.parse_args()

    if args.train:
        train_model()
    elif args.realtime:
        realtime_prediction()
    elif args.predict:
        predict_image(args.predict)
    else:
        print("Usage: python braille_vgg16.py [--train | --realtime | --predict <image_path>]")
