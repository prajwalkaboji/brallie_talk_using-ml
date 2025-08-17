import os
import cv2
import numpy as np
import collections
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# =========================
# CONFIG
# =========================
IMG_SIZE = 64
EPOCHS = 50
BATCH_SIZE = 32

TRAIN_SRC = r"D:\bml\num"      # Kaggle dataset
TEST_SRC = r"D:\bml\numbers"   # OpenCV dataset

# =========================
# DATA LOADER
# =========================
def load_images_from_folder(folder):
    images, labels = [], []
    class_names = sorted(os.listdir(folder))  # folder names = labels
    class_map = {name: idx for idx, name in enumerate(class_names)}

    for label in class_names:
        path = os.path.join(folder, label)
        if not os.path.isdir(path):
            continue
        for file in os.listdir(path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(path, file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(class_map[label])

    return np.array(images), np.array(labels), class_names

def prepare_dataset(train_path, test_path):
    print(f"[📦] Loading training data from: {train_path}")
    X_train, y_train, class_names = load_images_from_folder(train_path)
    print(f"[📦] Loading testing data from: {test_path}")
    X_test, y_test, _ = load_images_from_folder(test_path)

    # Normalize
    X_train, X_test = X_train / 255.0, X_test / 255.0

    num_classes = len(class_names)
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Print dataset distribution
    print("[📊] Training class distribution:", collections.Counter(np.argmax(y_train, axis=1)))
    print("[📊] Testing class distribution:", collections.Counter(np.argmax(y_test, axis=1)))

    return X_train, X_test, y_train, y_test, num_classes, class_names

# =========================
# MODEL
# =========================
def build_minivgg(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(32, (3,3), activation="relu", padding="same", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, num_classes, class_names = prepare_dataset(TRAIN_SRC, TEST_SRC)

    print(f"[✅] Detected {num_classes} classes: {class_names}")

    model = build_minivgg((IMG_SIZE, IMG_SIZE, 3), num_classes)
    model.summary()

    print("[🚀] Training model...")
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE)

    print("[💾] Saving model as 'minivgg_auto.h5'...")
    model.save("minivgg_auto.h5")

    print("[✅] Training complete. Evaluating model...")
    loss, acc = model.evaluate(X_test, y_test)
    print(f"[🎯] Test Accuracy: {acc * 100:.2f}%")
