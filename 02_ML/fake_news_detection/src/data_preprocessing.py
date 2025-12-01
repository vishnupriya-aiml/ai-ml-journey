"""
data_preprocessing.py
Preprocessing utilities for Fake News Detection:
- load data
- clean text
- TF-IDF vectorization
- train/test split
"""

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV file containing text and labels (FAKE / REAL).
    """
    df = pd.read_csv(path)
    return df


def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - lowercase
    - remove non alphabetic characters
    - collapse multiple spaces
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    # keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)
    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_data(path: str):
    """
    Full preprocessing pipeline:
    - load data
    - clean text
    - TF-IDF vectorization
    - train/test split

    Returns:
        X_train, X_test, y_train, y_test, vectorizer
    """
    df = load_data(path)

    # Clean text
    df["clean_text"] = df["text"].apply(clean_text)

    X_text = df["clean_text"]
    y = df["label"]

    # TF-IDF vectorizer for text features
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams
        max_features=5000,    # cap vocabulary size
        stop_words="english"  # remove common English stop words
    )

    X = vectorizer.fit_transform(X_text)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, vectorizer


if __name__ == "__main__":
    # Small self-test to verify shapes
    X_train, X_test, y_train, y_test, vectorizer = preprocess_data(
        "data/fake_news_small.csv"
    )
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Train labels distribution:\n", y_train.value_counts())
