"""
tokenizer_builder.py
Tokenizer and encoding utilities for BERT sentiment classifier.
"""

from typing import List, Tuple, Dict

import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 64  # max tokens per text


def get_tokenizer(model_name: str = MODEL_NAME):
    """
    Load a pre-trained BERT tokenizer from HuggingFace.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def encode_texts(
    tokenizer,
    texts: List[str],
    max_length: int = MAX_LENGTH,
) -> Dict[str, np.ndarray]:
    """
    Tokenize and encode a list of texts into input_ids and attention_mask.

    Returns:
        {
            "input_ids": np.ndarray of shape (N, max_length),
            "attention_mask": np.ndarray of shape (N, max_length),
        }
    """
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np",  # return numpy arrays
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def encode_labels(labels: List[str]) -> np.ndarray:
    """
    Encode 'negative' as 0 and 'positive' as 1.
    """
    mapping = {"negative": 0, "positive": 1}
    encoded = [mapping[label] for label in labels]
    return np.array(encoded, dtype="int32")


def prepare_bert_data(
    texts: List[str],
    labels: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
    max_length: int = MAX_LENGTH,
):
    """
    Full preprocessing pipeline for BERT:
    - load tokenizer
    - tokenize & encode texts
    - encode labels
    - split into train/test

    Returns:
        X_train (dict: input_ids, attention_mask),
        X_test (dict),
        y_train (np.ndarray),
        y_test (np.ndarray),
        tokenizer
    """
    tokenizer = get_tokenizer()

    encoded = encode_texts(tokenizer, texts, max_length=max_length)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    y = encode_labels(labels)

    X_train_ids, X_test_ids, X_train_mask, X_test_mask, y_train, y_test = train_test_split(
        input_ids,
        attention_mask,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    X_train = {
        "input_ids": X_train_ids,
        "attention_mask": X_train_mask,
    }
    X_test = {
        "input_ids": X_test_ids,
        "attention_mask": X_test_mask,
    }

    return X_train, X_test, y_train, y_test, tokenizer


if __name__ == "__main__":
    # Small self-test using the dataset
    from data_loader import load_bert_dataset

    texts, labels = load_bert_dataset()
    X_train, X_test, y_train, y_test, tokenizer = prepare_bert_data(texts, labels)

    print("Example text:", texts[0])
    encoded_example = tokenizer(texts[0], padding="max_length", truncation=True, max_length=MAX_LENGTH)
    print("Example input_ids (first 10):", encoded_example["input_ids"][:10])
    print("Train input_ids shape:", X_train["input_ids"].shape)
    print("Train attention_mask shape:", X_train["attention_mask"].shape)
    print("Train labels shape:", y_train.shape)
    print("Test input_ids shape:", X_test["input_ids"].shape)
    print("Test labels shape:", y_test.shape)
