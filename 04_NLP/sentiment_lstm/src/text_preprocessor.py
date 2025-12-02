"""
text_preprocessor.py
Tokenization, padding, and label encoding for sentiment LSTM project.
"""

from typing import List, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_VOCAB_SIZE = 5000   # max number of words in vocabulary
MAX_SEQUENCE_LENGTH = 30  # max tokens per sentence


def build_tokenizer(texts: List[str], num_words: int = MAX_VOCAB_SIZE) -> Tokenizer:
    """
    Fit a Keras Tokenizer on the given texts.
    """
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    return tokenizer


def texts_to_padded_sequences(
    tokenizer: Tokenizer,
    texts: List[str],
    max_len: int = MAX_SEQUENCE_LENGTH,
) -> np.ndarray:
    """
    Convert raw texts to sequences of integers and pad them to fixed length.
    """
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(
        sequences,
        maxlen=max_len,
        padding="post",
        truncating="post",
    )
    return padded


def encode_labels(labels: List[str]) -> np.ndarray:
    """
    Encode 'negative' as 0 and 'positive' as 1.
    """
    mapping = {"negative": 0, "positive": 1}
    encoded = [mapping[label] for label in labels]
    return np.array(encoded, dtype="int32")


def prepare_data(
    texts: List[str],
    labels: List[str],
    num_words: int = MAX_VOCAB_SIZE,
    max_len: int = MAX_SEQUENCE_LENGTH,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Full preprocessing pipeline:
    - build tokenizer
    - convert texts to padded sequences
    - encode labels
    - train/test split

    Returns:
        X_train, X_test, y_train, y_test, tokenizer
    """
    tokenizer = build_tokenizer(texts, num_words=num_words)
    X = texts_to_padded_sequences(tokenizer, texts, max_len=max_len)
    y = encode_labels(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, tokenizer


if __name__ == "__main__":
    # Small self-test using the existing dataset
    from data_loader import load_sentiment_data

    texts, labels = load_sentiment_data()
    X_train, X_test, y_train, y_test, tokenizer = prepare_data(texts, labels)

    print("Example text:", texts[0])
    print("Encoded sequence for first text:", tokenizer.texts_to_sequences([texts[0]]))
    print("Train shape:", X_train.shape, y_train.shape)
    print("Test shape:", X_test.shape, y_test.shape)
    print("Word index size:", len(tokenizer.word_index))
