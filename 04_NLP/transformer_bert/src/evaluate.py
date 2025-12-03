"""
evaluate.py
Evaluate the trained Transformer-style sentiment classifier and
generate classification report & confusion matrix.
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import load_bert_dataset
from tokenizer_builder import prepare_bert_data, MAX_LENGTH


def evaluate_transformer(
    model_path: str = "models/transformer_sentiment_best.keras",
    reports_dir: str = "reports",
):
    # 1. Load raw data
    texts, labels = load_bert_dataset()

    # 2. Encode and split (same config as training)
    X_train, X_test, y_train, y_test, tokenizer = prepare_bert_data(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        max_length=MAX_LENGTH,
    )

    # 3. Check and load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = tf.keras.models.load_model(model_path)
    print(f"Loaded model from: {model_path}")

    # 4. Evaluate on test set
    test_loss, test_acc = model.evaluate(
        X_test["input_ids"],
        y_test,
        verbose=1,
    )
    print(f"Test accuracy: {test_acc * 100:.2f}%")

    # 5. Predictions
    y_prob = model.predict(X_test["input_ids"], verbose=1)
    # y_prob shape: (N, 2); take argmax to get predicted class 0/1
    y_pred = np.argmax(y_prob, axis=1)

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

    # 7. Save report to file
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
    evaluate_transformer()
