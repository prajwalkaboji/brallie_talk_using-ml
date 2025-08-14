import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add, Input, Flatten, Dense, MaxPooling2D, Dropout
from tensorflow.keras.models import Model

# ResNet block with projection option
def resnet_block(x, filters, kernel_size=3, use_projection=False):
    shortcut = x

    if use_projection:  # match dimensions if needed
        shortcut = Conv2D(filters, (1, 1), padding="same")(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Conv2D(filters, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)

    x = Add()([shortcut, x])
    x = Activation("relu")(x)
    return x

# Build model
input_layer = Input(shape=(28, 28, 1))

x = Conv2D(32, (3, 3), padding="same", activation="relu")(input_layer)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = resnet_block(x, 32)
x = resnet_block(x, 64, use_projection=True)  # projection since filters change
x = MaxPooling2D((2, 2))(x)
x = resnet_block(x, 64)

x = Flatten()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output_layer = Dense(10, activation="softmax")(x)  # Change '10' to number of classes

model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()
