"""
model_builder.py
Defines a Transformer-style sentiment classifier using pure
TensorFlow / Keras layers.

We still use a BERT-style tokenizer (WordPiece IDs), but instead of
loading the full HuggingFace TFBertModel (which can conflict with some
TF/Keras setups), we build a lightweight Transformer encoder using
MultiHeadAttention.

This is simpler, more stable locally, and still counts as a
Transformer-based text classifier.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

from tokenizer_builder import MAX_LENGTH

# Rough BERT vocab size (bert-base-uncased)
VOCAB_SIZE = 30522
EMBED_DIM = 128
NUM_HEADS = 4
FF_DIM = 256  # feed-forward layer size
NUM_LABELS = 2


def transformer_block(x):
    """
    A simple Transformer encoder block:
    - Multi-head self attention
    - Add & LayerNorm
    - Feed-forward network
    - Add & LayerNorm
    """
    # Self-attention
    attn_output = layers.MultiHeadAttention(
        num_heads=NUM_HEADS, key_dim=EMBED_DIM
    )(x, x)
    x = layers.Add()([x, attn_output])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Feed-forward
    ff_output = layers.Dense(FF_DIM, activation="relu")(x)
    ff_output = layers.Dense(EMBED_DIM)(ff_output)
    x = layers.Add()([x, ff_output])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    return x


def build_transformer_classifier(
    vocab_size: int = VOCAB_SIZE,
    max_len: int = MAX_LENGTH,
    num_labels: int = NUM_LABELS,
    learning_rate: float = 2e-4,
) -> tf.keras.Model:
    """
    Build and compile a simple Transformer-based classifier:

    - Token IDs input
    - Word embeddings
    - Positional embeddings
    - Transformer encoder block
    - Global average pooling
    - Dense classifier head
    """

    # Inputs
    input_ids = layers.Input(shape=(max_len,), dtype="int32", name="input_ids")

    # Token embeddings
    token_emb = layers.Embedding(
        input_dim=vocab_size,
        output_dim=EMBED_DIM,
        name="token_embedding",
    )(input_ids)

    # Positional embeddings
    positions = tf.range(start=0, limit=max_len, delta=1)
    pos_emb_layer = layers.Embedding(
        input_dim=max_len,
        output_dim=EMBED_DIM,
        name="position_embedding",
    )
    pos_emb = pos_emb_layer(positions)

    # Add token + positional embeddings
    x = token_emb + pos_emb

    # Transformer encoder block
    x = transformer_block(x)

    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    # Output
    outputs = layers.Dense(num_labels, activation="softmax")(x)

    model = Model(inputs=input_ids, outputs=outputs, name="transformer_sentiment_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    # Self-test: build and show summary
    model = build_transformer_classifier()
    model.summary()
