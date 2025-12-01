"""
evaluate_model.py
Evaluate the saved fake news detection model on the full dataset.
"""

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from data_preprocessing import clean_text


def evaluate():
    # Load trained model + vectorizer
    model = joblib.load("models/fake_news_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")

    # Load original dataset
    df = pd.read_csv("data/fake_news_small.csv")

    # Clean text using the same function
    df["clean_text"] = df["text"].apply(clean_text)

    # Transform text to TF-IDF
    X_all = vectorizer.transform(df["clean_text"])
    y_true = df["label"]

    # Predictions
    y_pred = model.predict(X_all)

    # Metrics
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(report)

    # Save evaluation report to file
    with open("reports/evaluation_report.txt", "w") as f:
        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification Report:\n")
        f.write(report)

    print("\nEvaluation report saved to reports/evaluation_report.txt")


if __name__ == "__main__":
    evaluate()
