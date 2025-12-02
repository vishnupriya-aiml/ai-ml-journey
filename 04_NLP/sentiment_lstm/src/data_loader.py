"""
data_loader.py
Loads the sentiment dataset from CSV.
"""

import os
import pandas as pd
from typing import Tuple


def load_sentiment_data(csv_path: str = "data/reviews_lstm.csv") -> Tuple[list, list]:
    """
    Load the sentiment dataset.

    Returns:
    - texts: list of raw text strings
    - labels: list of 'positive' or 'negative'
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    texts = df["text"].tolist()
    labels = df["label"].tolist()

    return texts, labels


if __name__ == "__main__":
    texts, labels = load_sentiment_data()
    print("Loaded texts:", texts)
    print("Loaded labels:", labels)
