"""
data_loader.py
Loads sentiment dataset for BERT project.
"""

import os
import pandas as pd


def load_bert_dataset(csv_path="data/sentiment_bert.csv"):
    """
    Load CSV file containing text + label.
    Returns:
        texts: list[str]
        labels: list[str]
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    return texts, labels


if __name__ == "__main__":
    texts, labels = load_bert_dataset()
    print("Sample text:", texts[0])
    print("Sample label:", labels[0])
