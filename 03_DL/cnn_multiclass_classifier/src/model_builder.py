"""
model_builder.py — Multi-class CNN image classifier
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def build_cnn_classifier(
    input_shape=(32, 32, 3),
    num_classes=3,
    learning_rate=0.001,
) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape, name="image_input")

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="cnn_multiclass_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    model = build_cnn_classifier()
    model.summary()
