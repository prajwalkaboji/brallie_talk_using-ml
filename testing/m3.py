import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# ============================
# CONFIGURATION
# ============================
DATASET_DIR = "E:/bml/BrailleDataset"
MODEL_PATH = r"E:\bml\braille_character_model_alexnet.h5"
IMG_SIZE = (28, 28)
EPOCHS = 15
BATCH_SIZE = 32

# ============================
# LOAD DATA
# ============================
X = []
y = []

label_map = {chr(i+65): i for i in range(26)}
reverse_map = {v: k for k, v in label_map.items()}

for filename in os.listdir(DATASET_DIR):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        label_char = filename[0].upper()
        if label_char in label_map:
            img_path = os.path.join(DATASET_DIR, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, IMG_SIZE)
            X.append(img)
            y.append(label_map[label_char])

X = np.array(X, dtype="float32") / 255.0
X = np.expand_dims(X, axis=-1)
y = np.array(y)
y = to_categorical(y, num_classes=26)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

# ============================
# ALEXNET-LIKE MODEL
# ============================
model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1), padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(256, (3,3), activation='relu', padding='same'),
    BatchNormalization(),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(26, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

# ============================
# TRAIN
# ============================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# ============================
# EVALUATION
# ============================
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_acc*100:.2f}%")

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc*100:.2f}%")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=list(label_map.keys())))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true_classes, y_pred_classes))

# ============================
# PREDICTION FUNCTION
# ============================
def predict_braille_image(image_path):
    model = load_model(MODEL_PATH)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    return reverse_map[predicted_class]

# Example usage
test_img_path = "test_braille_char.png"
if os.path.exists(test_img_path):
    print("Predicted Character:", predict_braille_image(test_img_path))
