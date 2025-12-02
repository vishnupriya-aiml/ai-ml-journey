"""
model_builder.py
Defines an LSTM-based model for sentiment classification
using TensorFlow / Keras.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

from text_preprocessor import MAX_VOCAB_SIZE, MAX_SEQUENCE_LENGTH


def build_lstm_model(
    vocab_size: int = MAX_VOCAB_SIZE,
    embedding_dim: int = 64,
    lstm_units: int = 64,
    max_len: int = MAX_SEQUENCE_LENGTH,
) -> tf.keras.Model:
    """
    Build and compile an LSTM model for binary sentiment classification.

    Architecture:
    - Embedding
    - LSTM
    - Dense + sigmoid
    """

    # +1 for OOV token index
    model = models.Sequential()
    model.add(
        layers.Embedding(
            input_dim=vocab_size + 1,
            output_dim=embedding_dim,
            input_length=max_len,
        )
    )
    model.add(layers.LSTM(lstm_units))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation="sigmoid"))  # binary classification

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    # Self-test: build the model and print summary
    lstm_model = build_lstm_model()
    lstm_model.summary()
