"""
evaluate.py
Evaluate the saved CNN model on the CIFAR-10 test set and
generate metrics (accuracy, classification report, confusion matrix).
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import load_cifar10, get_class_names


def evaluate_model(
    model_path: str = "models/cnn_cifar10_best.keras",
    reports_dir: str = "reports",
):
    # Load test data
    (_, _), (x_test, y_test) = load_cifar10()

    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = tf.keras.models.load_model(model_path)
    print(f"Loaded model from: {model_path}")

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print(f"Test accuracy: {test_acc * 100:.2f}%")

    # Predictions
    y_prob = model.predict(x_test, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    # Metrics
    class_names = get_class_names()
    report = classification_report(
        y_test, y_pred, target_names=class_names, digits=4
    )
    cm = confusion_matrix(y_test, y_pred)

    print("\nClassification Report:")
    print(report)

    print("\nConfusion Matrix:")
    print(cm)

    # Ensure reports directory exists
    os.makedirs(reports_dir, exist_ok=True)

    # Save evaluation report to file
    report_path = os.path.join(reports_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test accuracy: {test_acc * 100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))

    print(f"\nEvaluation report saved to: {report_path}")


if __name__ == "__main__":
    evaluate_model()
