"""
model_builder.py — Bidirectional LSTM classifier model
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

from tokenizer_builder import MAX_LENGTH, MAX_WORDS


def build_bilstm_classifier(
    vocab_size: int = MAX_WORDS,
    embed_dim: int = 64,
    lstm_units: int = 64,
    num_labels: int = 2,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    inputs = layers.Input(shape=(MAX_LENGTH,), dtype="int32", name="input_ids")

    x = layers.Embedding(vocab_size, embed_dim, name="embedding")(inputs)

    # Bidirectional LSTM
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=False),
        name="bilstm",
    )(x)

    x = layers.Dense(32, activation="relu", name="dense_1")(x)
    x = layers.Dropout(0.3, name="dropout")(x)

    outputs = layers.Dense(num_labels, activation="softmax", name="output")(x)

    model = Model(inputs, outputs, name="bilstm_sentiment_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    model = build_bilstm_classifier()
    model.summary()
