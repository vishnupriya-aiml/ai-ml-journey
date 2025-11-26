"""
data_preprocessing.py
Utility functions for loading and preparing text data for sentiment analysis.
"""

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


def load_data(path: str) -> pd.DataFrame:
    """
    Load the CSV file containing text and sentiment labels.
    """
    df = pd.read_csv(path)
    return df


def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - lowercase
    - remove punctuation
    - remove extra spaces
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    # remove anything not a-z or space
    text = re.sub(r"[^a-z\s]", "", text)
    # collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_data(path: str):
    """
    Load the data, clean the text, vectorize it, and split into train/test sets.
    Returns:
        X_train, X_test, y_train, y_test, vectorizer
    """
    df = load_data(path)

    # Clean the text column
    df["clean_text"] = df["text"].apply(clean_text)

    # Features (text) and labels (sentiment)
    X_text = df["clean_text"]
    y = df["sentiment"]

    # Convert text to bag-of-words using CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X_text)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    return X_train, X_test, y_train, y_test, vectorizer
