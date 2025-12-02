"""
model_builder.py
Defines a Convolutional Neural Network (CNN) model for CIFAR-10
using TensorFlow / Keras.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_model(input_shape=(32, 32, 3), num_classes=10) -> tf.keras.Model:
    """
    Build and return a simple CNN model for image classification.

    Architecture:
    - Conv2D + ReLU + MaxPool
    - Conv2D + ReLU + MaxPool
    - Conv2D + ReLU + MaxPool
    - Flatten
    - Dense + ReLU
    - Dropout
    - Output Dense with softmax
    """
    model = models.Sequential()

    # Block 1
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 2
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 3
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten + Dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation="softmax"))

    # Compile the model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    # Small self-test: build model and print summary
    cnn_model = build_cnn_model()
    cnn_model.summary()
