"""
tokenizer_builder.py — prepares tokenizer + padded sequences for BiLSTM model
"""

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


MAX_WORDS = 5000
MAX_LENGTH = 50


def prepare_bilstm_data(texts, labels, test_size=0.2, random_state=42):
    """
    - Fit tokenizer on texts
    - Convert texts to sequences and pad
    - Convert labels to NumPy array (required for validation_split)
    - Train/test split
    """
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding="post")

    labels_array = np.array(labels, dtype="int32")

    X_train, X_test, y_train, y_test = train_test_split(
        padded,
        labels_array,
        test_size=test_size,
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test, tokenizer


if __name__ == "__main__":
    # tiny self-test
    sample_texts = ["I love this", "I hate this"]
    sample_labels = [1, 0]
    X_train, X_test, y_train, y_test, tok = prepare_bilstm_data(sample_texts, sample_labels)
    print("X_train shape:", X_train.shape)
    print("y_train:", y_train)
