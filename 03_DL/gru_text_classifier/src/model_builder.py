"""
model_builder.py — GRU classifier model
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

from tokenizer_builder import MAX_LENGTH


def build_gru_classifier(
    vocab_size=5000,
    embed_dim=64,
    gru_units=64,
    num_labels=2,
    learning_rate=0.001,
):
    inputs = layers.Input(shape=(MAX_LENGTH,), dtype="int32")

    x = layers.Embedding(vocab_size, embed_dim)(inputs)

    x = layers.GRU(gru_units, return_sequences=False)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_labels, activation="softmax")(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    model = build_gru_classifier()
    model.summary()
