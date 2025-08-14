import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input, Add, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# =========================
# CONFIGURATION
# =========================
DATASET_DIR = "E:/bml/BrailleDataset"
MODEL_PATH = r"E:\bml\braille_resnet_model.h5"
IMG_SIZE = (28, 28)
EPOCHS = 15
BATCH_SIZE = 32

# =========================
# LOAD DATA
# =========================
X = []
y = []

label_map = {chr(i+65): i for i in range(26)}
reverse_map = {v: k for k, v in label_map.items()}

for filename in os.listdir(DATASET_DIR):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
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

# =========================
# RESNET BLOCK FUNCTION
# =========================
def resnet_block(x, filters, kernel_size=3):
    # Shortcut path to match dimensions
    shortcut = Conv2D(filters, (1, 1), padding="same")(x)
    shortcut = BatchNormalization()(shortcut)

    # Main path
    x = Conv2D(filters, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)

    # Add & activate
    x = Add()([shortcut, x])
    x = Activation("relu")(x)
    return x


# =========================
# MODEL
# =========================
inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1))
x = Conv2D(32, (3,3), padding="same", activation="relu")(inputs)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)

x = resnet_block(x, 32)
x = MaxPooling2D((2,2))(x)

x = resnet_block(x, 64)
x = MaxPooling2D((2,2))(x)

x = Flatten()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)
outputs = Dense(26, activation="softmax")(x)

model = Model(inputs, outputs)
model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

# =========================
# TRAIN
# =========================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# Save model
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# =========================
# EVALUATION
# =========================
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

# =========================
# PREDICTION FUNCTION
# =========================
def predict_braille_image(image_path):
    model = tf.keras.models.load_model(MODEL_PATH)
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
