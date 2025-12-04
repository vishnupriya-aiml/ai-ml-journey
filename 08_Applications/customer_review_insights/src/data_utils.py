"""
data_utils.py
Utilities to load review datasets and attach sentiment predictions.
"""

import os
import pandas as pd

from src.model_utils import predict_sentiments


def load_reviews_csv(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    df = pd.read_csv(path)
    return df


def prepare_reviews_dataframe(df: pd.DataFrame):
    """
    Ensure expected columns exist and clean the text field.
    Expected columns: 'review_text', 'rating' (rating is optional for predictions).
    """
    if "review_text" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'review_text' column.")

    df = df.copy()
    df["review_text"] = df["review_text"].astype(str).fillna("")

    return df


def add_sentiment_predictions(df: pd.DataFrame):
    """
    Adds:
    - sentiment_label: 'positive' / 'negative'
    - sentiment_score: max probability for predicted class
    """
    df = prepare_reviews_dataframe(df)
    texts = df["review_text"].tolist()

    result = predict_sentiments(texts)
    pred_labels = result["pred_label"]
    probs = result["probs"]

    max_scores = probs.max(axis=1)

    df["sentiment_label"] = pred_labels
    df["sentiment_score"] = max_scores

    return df
