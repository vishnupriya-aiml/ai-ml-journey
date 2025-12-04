"""
evaluate.py — Evaluate multi-class CNN classifier
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import load_cifar_subset


def evaluate_cnn_multiclass(
    model_path: str = "models/cnn_multiclass_best.keras",
    reports_dir: str = "reports",
):
    # 1. Load data
    X_train, X_test, y_train, y_test, class_names = load_cifar_subset()

    # 2. Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = tf.keras.models.load_model(model_path)
    print(f"Loaded model from: {model_path}")

    # 3. Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    # 4. Predictions
    y_prob = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    # 5. Classification report + confusion matrix
    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        digits=4,
    )
    cm = confusion_matrix(y_test, y_pred)

    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)

    # 6. Save report
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy: {test_acc * 100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))

    print(f"\nEvaluation report saved to: {report_path}")


if __name__ == "__main__":
    evaluate_cnn_multiclass()
