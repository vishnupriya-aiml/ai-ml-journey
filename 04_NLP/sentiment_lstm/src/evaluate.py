"""
evaluate.py
Evaluate the saved LSTM model on the sentiment test set and
generate classification report & confusion matrix.
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import load_sentiment_data
from text_preprocessor import prepare_data, MAX_VOCAB_SIZE, MAX_SEQUENCE_LENGTH


def evaluate_lstm(
    model_path: str = "models/lstm_sentiment_best.keras",
    reports_dir: str = "reports",
):
    # 1. Load raw data
    texts, labels = load_sentiment_data()

    # 2. Preprocess to get train/test (same split as training)
    X_train, X_test, y_train, y_test, tokenizer = prepare_data(
        texts,
        labels,
        num_words=MAX_VOCAB_SIZE,
        max_len=MAX_SEQUENCE_LENGTH,
        test_size=0.2,
        random_state=42,
    )

    # 3. Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = tf.keras.models.load_model(model_path)
    print(f"Loaded model from: {model_path}")

    # 4. Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test accuracy: {test_acc * 100:.2f}%")

    # 5. Predictions
    y_prob = model.predict(X_test, verbose=1)
    # For sigmoid output, threshold at 0.5
    y_pred = (y_prob.flatten() >= 0.5).astype(int)

    # 6. Classification report & confusion matrix
    target_names = ["negative", "positive"]
    report = classification_report(
        y_test, y_pred, target_names=target_names, digits=4
    )
    cm = confusion_matrix(y_test, y_pred)

    print("\nClassification Report:")
    print(report)

    print("\nConfusion Matrix:")
    print(cm)

    # 7. Save to file
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test accuracy: {test_acc * 100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))

    print(f"\nEvaluation report saved to: {report_path}")


if __name__ == "__main__":
    evaluate_lstm()
