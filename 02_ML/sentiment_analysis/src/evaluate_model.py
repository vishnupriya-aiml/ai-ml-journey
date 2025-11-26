"""
evaluate_model.py
Load the trained sentiment model and print evaluation metrics on the full dataset.
"""

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from data_preprocessing import clean_text


def evaluate():
    # Load model and vectorizer
    model = joblib.load("models/sentiment_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")

    # Load original data
    df = pd.read_csv("data/reviews_small.csv")

    # Clean text using same function
    df["clean_text"] = df["text"].apply(clean_text)

    # Transform text with loaded vectorizer
    X_all = vectorizer.transform(df["clean_text"])
    y_true = df["sentiment"]

    # Predictions
    y_pred = model.predict(X_all)

    # Metrics
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    evaluate()
