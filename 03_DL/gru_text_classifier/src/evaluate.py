"""
evaluate.py — Evaluate GRU classifier
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import load_gru_dataset
from tokenizer_builder import prepare_gru_data


def evaluate_gru(model_path="models/gru_sentiment_best.keras", reports_dir="reports"):
    texts, labels = load_gru_dataset()

    X_train, X_test, y_train, y_test, tokenizer = prepare_gru_data(texts, labels)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = tf.keras.models.load_model(model_path)
    print(f"Loaded model from {model_path}")

    loss, acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Accuracy: {acc:.4f}")

    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    os.makedirs(reports_dir, exist_ok=True)

    report_path = os.path.join(reports_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    print(f"Evaluation saved to {report_path}")


if __name__ == "__main__":
    evaluate_gru()
