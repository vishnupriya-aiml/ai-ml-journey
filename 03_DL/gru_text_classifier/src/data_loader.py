"""
data_loader.py — loads the small GRU sentiment dataset
"""

import pandas as pd
import os


def load_gru_dataset(path="data/sentiment_gru.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)

    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    return texts, labels
