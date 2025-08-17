import os
import cv2
import numpy as np
import argparse
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

IMG_SIZE = 64
DATASET_PATH = "dataset"
MODEL_NAME = "minivgg_1to10.keras"

# -------------------------------
# 1. Dataset Generator
# -------------------------------
def make_dataset(save_path=DATASET_PATH, samples_per_class=500, img_size=64):
    os.makedirs(save_path, exist_ok=True)

    for digit in range(1, 11):
        digit_path = os.path.join(save_path, str(digit).zfill(2))  # 01,02,...,10
        os.makedirs(digit_path, exist_ok=True)

        for i in range(samples_per_class):
            img = np.zeros((img_size, img_size), dtype=np.uint8)

            spacing = img_size // 5
            for d in range(digit):
                center = (spacing + (d % 5) * spacing, spacing + (d // 5) * spacing)
                cv2.circle(img, center, img_size // 12, 255, -1)

            # Blur
            img = cv2.GaussianBlur(img, (3, 3), 0)

            # ✅ Adaptive threshold
            img = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            # ✅ Add random Gaussian noise
            noise = np.random.normal(0, 25, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            img = cv2.resize(img, (img_size, img_size))
            cv2.imwrite(os.path.join(digit_path, f"{i}.png"), img)

    print("[✅] Dataset generated at", save_path)


# -------------------------------
# 2. Load Dataset
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
            img = img.astype("float32") / 255.0
            data.append(img)
            labels.append(idx)

    data = np.expand_dims(np.array(data), axis=-1)
    labels = to_categorical(np.array(labels), num_classes=len(classes))
    return data, labels


# -------------------------------
# 3. MiniVGG Model
# -------------------------------
def build_minivgg(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=10):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# -------------------------------
# 4. Training
# -------------------------------
def train_model():
    print("[📦] Loading dataset...")
    X, y = load_dataset(DATASET_PATH, IMG_SIZE)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[🧠] Building MiniVGG...")
    model = build_minivgg((IMG_SIZE, IMG_SIZE, 1), 10)

    print("[🚀] Training...")
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=32)

    model.save(MODEL_NAME)
    print(f"[💾] Model saved as {MODEL_NAME}")

    loss, acc = model.evaluate(X_test, y_test)
    print(f"[🎯] Test Accuracy: {acc*100:.2f}%")


# -------------------------------
# 5. Realtime Webcam Prediction
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

        # ROI box
        x, y, w, h = 100, 100, 200, 200
        roi = frame[y:y+h, x:x+w]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # ✅ Same adaptive threshold
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        gray = gray.astype("float32") / 255.0
        gray = np.expand_dims(gray, axis=-1)
        gray = np.expand_dims(gray, axis=0)

        pred = model.predict(gray, verbose=0)
        label = np.argmax(pred) + 1  # class 01 → digit 1
        conf = np.max(pred)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Num: {label} ({conf*100:.1f}%)",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Realtime Braille Num Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# -------------------------------
# 6. Main
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--make_data", action="store_true", help="Generate dataset")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--realtime", action="store_true", help="Run realtime webcam")
    args = parser.parse_args()

    if args.make_data:
        make_dataset()
    elif args.train:
        train_model()
    elif args.realtime:
        realtime_prediction()
    else:
        print("Usage: python braille_num.py [--make_data | --train | --realtime]")
