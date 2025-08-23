import os
import cv2
import numpy as np
import argparse
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

IMG_SIZE = 64
DATASET_PATH = r"D:\bml\mini1"
MODEL_NAME = "vgg16_braille.keras"

# -------------------------------
# 1. Load Dataset
# -------------------------------
def load_dataset(path=DATASET_PATH, img_size=64):
    data, labels = [], []
    classes = sorted(os.listdir(path))

    for idx, c in enumerate(classes):
        class_dir = os.path.join(path, c)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_size, img_size))
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # ✅ VGG expects 3 channels
            img = img.astype("float32") / 255.0
            data.append(img)
            labels.append(idx)

    data = np.array(data)
    labels = to_categorical(np.array(labels), num_classes=len(classes))
    return data, labels


# -------------------------------
# 2. Build VGG16 Model
# -------------------------------
def build_vgg16(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=5):
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
    X, y = load_dataset(DATASET_PATH, IMG_SIZE)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[🧠] Building VGG16...")
    model = build_vgg16((IMG_SIZE, IMG_SIZE, 3), 5)

    print("[🚀] Training...")
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

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
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)  # ✅ VGG expects 3 channels
        gray = gray.astype("float32") / 255.0
        gray = np.expand_dims(gray, axis=0)

        pred = model.predict(gray, verbose=0)
        label = np.argmax(pred) + 1
        conf = np.max(pred)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Num: {label} ({conf*100:.1f}%)",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Realtime Braille Num Recognition (VGG16)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# -------------------------------
# 5. Main
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--realtime", action="store_true", help="Run realtime webcam")
    args = parser.parse_args()

    if args.train:
        train_model()
    elif args.realtime:
        realtime_prediction()
    else:
        print("Usage: python braille_vgg16.py [--train | --realtime]")
