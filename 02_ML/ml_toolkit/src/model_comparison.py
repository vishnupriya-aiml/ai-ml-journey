"""
model_comparison.py
Compare multiple ML models on the same preprocessed dataset.

Models:
- Logistic Regression
- Random Forest
- Gradient Boosting

Outputs:
- Accuracy scores for each model
- Classification report for the best model
- Confusion matrix for the best model
- Written summary in reports/model_comparison.txt
"""

import os
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocessing_template import preprocess_dataset


def run_model_comparison(
    data_path: str = os.path.join("data", "students_sample.csv"),
    target_col: str = "passed",
):
    # Define feature types for preprocessing
    numeric_features = ["hours_studied", "attendance_rate"]
    categorical_features = ["gender", "school_type"]

    # 1. Preprocess
    X_train, X_test, y_train, y_test, preprocessor = preprocess_dataset(
        csv_path=data_path,
        target_col=target_col,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        test_size=0.3,
        random_state=42,
    )

    # 2. Define models
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }

    results = {}

    # 3. Train & evaluate each model
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {
            "model": model,
            "accuracy": acc,
            "y_pred": y_pred,
        }
        print(f"{name} accuracy: {acc:.4f}")

    # 4. Pick best model by accuracy
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_info = results[best_name]

    best_model = best_info["model"]
    best_accuracy = best_info["accuracy"]
    best_y_pred = best_info["y_pred"]

    print(f"\nBest model: {best_name} with accuracy {best_accuracy:.4f}")

    # 5. Classification report & confusion matrix for best model
    report = classification_report(y_test, best_y_pred, digits=4)
    cm = confusion_matrix(y_test, best_y_pred)

    print("\nClassification Report (best model):")
    print(report)
    print("\nConfusion Matrix (best model):")
    print(cm)

    # 6. Save to reports file
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "model_comparison.txt")

    with open(report_path, "w") as f:
        f.write("Model comparison on students_sample.csv\n\n")
        for name, info in results.items():
            f.write(f"{name}: accuracy = {info['accuracy']:.4f}\n")

        f.write(f"\nBest model: {best_name} (accuracy = {best_accuracy:.4f})\n\n")
        f.write("Classification Report (best model):\n")
        f.write(report)
        f.write("\n\nConfusion Matrix (best model):\n")
        f.write(str(cm))

    print(f"\nModel comparison report written to: {report_path}")


if __name__ == "__main__":
    run_model_comparison()
