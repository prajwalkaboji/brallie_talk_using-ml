import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==============================
# CONFIGURATION
# ==============================
DATASET_PATH = r"D:\bml\realtime"              # Your original dataset
OUTPUT_PATH = r"D:\bml\realtime_augmented"     # Where augmented images will be saved
IMG_SIZE = (64, 64)                            # Resize target
NUM_AUGMENTED_IMAGES = 50                      # How many new images per original

# ==============================
# AUGMENTATION SETUP
# ==============================
datagen = ImageDataGenerator(
    rotation_range=15,       # random rotation
    width_shift_range=0.1,   # horizontal shift
    height_shift_range=0.1,  # vertical shift
    zoom_range=0.2,          # zoom in/out
    shear_range=0.1,         # shearing
    brightness_range=[0.8, 1.2], # brightness variation
    horizontal_flip=False,   # Braille shouldn’t flip horizontally
    fill_mode="nearest"
)

# ==============================
# AUGMENT DATASET
# ==============================
os.makedirs(OUTPUT_PATH, exist_ok=True)

for label in os.listdir(DATASET_PATH):
    class_dir = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(class_dir):
        continue

    # Create output class folder
    output_class_dir = os.path.join(OUTPUT_PATH, label)
    os.makedirs(output_class_dir, exist_ok=True)

    print(f"[🔄] Augmenting class: {label}")

    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert to 3 channels
        img = np.expand_dims(img, axis=0)  # shape: (1, h, w, 3)

        # Generate augmented images
        aug_iter = datagen.flow(img, batch_size=1,
                                save_to_dir=output_class_dir,
                                save_prefix="aug",
                                save_format="jpg")

        for i in range(NUM_AUGMENTED_IMAGES):
            next(aug_iter)  # generates and saves new image

print(f"[✅] Augmented dataset created at: {OUTPUT_PATH}")
