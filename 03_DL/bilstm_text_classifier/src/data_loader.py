"""
data_loader.py — loads the small BiLSTM sentiment dataset
"""

import pandas as pd
import os


def load_bilstm_dataset(path="data/sentiment_bilstm.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)

    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    return texts, labels


if __name__ == "__main__":
    texts, labels = load_bilstm_dataset()
    print("Sample text:", texts[0])
    print("Sample label:", labels[0])
