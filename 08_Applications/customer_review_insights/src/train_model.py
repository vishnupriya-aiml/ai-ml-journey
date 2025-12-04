"""
train_model.py
Train a sentiment model on customer reviews using TF-IDF + Logistic Regression.

This script:
- Loads data from data/reviews_sample.csv
- Creates a binary sentiment label from rating (>=4 = positive, <=2 = negative, 3 = neutral/dropped)
- Splits into train/test
- Builds a pipeline: TF-IDF -> LogisticRegression
- Evaluates on the test set
- Saves model + vectorizer to models/
- Writes a training report to reports/training_report.txt
"""

import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


DATA_PATH = os.path.join("data", "reviews_sample.csv")
MODELS_DIR = "models"
REPORTS_DIR = "reports"


def load_and_prepare_data(csv_path=DATA_PATH):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Basic cleaning: drop rows with missing review_text or rating
    df = df.dropna(subset=["review_text", "rating"])

    # Create binary sentiment:
    # rating >= 4 -> 1 (positive)
    # rating <= 2 -> 0 (negative)
    # rating == 3 -> neutral -> drop for now
    df = df[df["rating"] != 3]
    df["sentiment"] = (df["rating"] >= 4).astype(int)

    texts = df["review_text"].astype(str).tolist()
    labels = df["sentiment"].astype(int).tolist()

    return texts, labels


def train_sentiment_model():
    # 1. Load data
    texts, labels = load_and_prepare_data(DATA_PATH)

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.3,
        random_state=42,
        stratify=labels,
    )

    # 3. TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 4. Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    # 5. Evaluate
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)

    # 6. Ensure directories exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 7. Save model and vectorizer
    model_path = os.path.join(MODELS_DIR, "sentiment_model.pkl")
    vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)

    print(f"\nModel saved to: {model_path}")
    print(f"Vectorizer saved to: {vec_path}")

    # 8. Save training report
    report_path = os.path.join(REPORTS_DIR, "training_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))

    print(f"Training report written to: {report_path}")


if __name__ == "__main__":
    train_sentiment_model()
