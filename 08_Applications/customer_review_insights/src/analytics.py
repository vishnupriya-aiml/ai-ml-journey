"""
analytics.py
Compute summary statistics and aggregates from predicted reviews DataFrame.
"""

import pandas as pd


def compute_sentiment_summary(df: pd.DataFrame):
    """
    Assumes df has column 'sentiment_label' with values 'positive' or 'negative'.
    Returns dict with total, counts, percentages.
    """
    total = len(df)
    if total == 0:
        return {
            "total_reviews": 0,
            "positive_count": 0,
            "negative_count": 0,
            "positive_pct": 0.0,
            "negative_pct": 0.0,
        }

    counts = df["sentiment_label"].value_counts().to_dict()
    pos = counts.get("positive", 0)
    neg = counts.get("negative", 0)

    pos_pct = (pos / total) * 100.0
    neg_pct = (neg / total) * 100.0

    return {
        "total_reviews": total,
        "positive_count": pos,
        "negative_count": neg,
        "positive_pct": pos_pct,
        "negative_pct": neg_pct,
    }


def sentiment_by_source(df: pd.DataFrame):
    """
    If df has 'source' column, compute sentiment distribution per source.
    Returns a DataFrame with counts per source & sentiment.
    """
    if "source" not in df.columns:
        return None

    pivot = (
        df.pivot_table(
            index="source",
            columns="sentiment_label",
            values="review_text",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )

    # Ensure both columns exist
    if "negative" not in pivot.columns:
        pivot["negative"] = 0
    if "positive" not in pivot.columns:
        pivot["positive"] = 0

    pivot["total"] = pivot["negative"] + pivot["positive"]

    return pivot
