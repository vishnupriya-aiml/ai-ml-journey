"""
hyperparameter_search.py
Hyperparameter tuning example using the ML toolkit preprocessing.

What it does:
- Uses the same students_sample.csv dataset
- Preprocesses with scaling + one-hot encoding
- Runs GridSearchCV on RandomForestClassifier
- Prints best params, best accuracy
- Saves a detailed report to reports/hyperparameter_search.txt
"""

import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from preprocessing_template import preprocess_dataset


def run_hyperparameter_search(
    data_path: str = os.path.join("data", "students_sample.csv"),
    target_col: str = "passed",
):
    # 1. Define feature types
    numeric_features = ["hours_studied", "attendance_rate"]
    categorical_features = ["gender", "school_type"]

    # 2. Preprocess dataset
    X_train, X_test, y_train, y_test, preprocessor = preprocess_dataset(
        csv_path=data_path,
        target_col=target_col,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        test_size=0.3,
        random_state=42,
    )

    # 3. Define a base model
    base_model = RandomForestClassifier(random_state=42)

    # 4. Define hyperparameter grid
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 3, 5],
        "min_samples_split": [2, 4],
    }

    # 5. GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Evaluate best model on test set
    test_accuracy = best_model.score(X_test, y_test)

    print("\nBest parameters from GridSearchCV:")
    print(best_params)
    print(f"Best CV accuracy: {best_score:.4f}")
    print(f"Test accuracy with best params: {test_accuracy:.4f}")

    # 6. Save report
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "hyperparameter_search.txt")

    with open(report_path, "w") as f:
        f.write("Hyperparameter search for RandomForest on students_sample.csv\n\n")
        f.write(f"Best parameters: {best_params}\n")
        f.write(f"Best CV accuracy: {best_score:.4f}\n")
        f.write(f"Test accuracy (best model): {test_accuracy:.4f}\n")
        f.write("\nFull grid results:\n")

        means = grid_search.cv_results_["mean_test_score"]
        stds = grid_search.cv_results_["std_test_score"]
        params = grid_search.cv_results_["params"]

        for mean, std, param in zip(means, stds, params):
            f.write(f"{mean:.4f} (+/- {std * 2:.4f}) for {param}\n")

    print(f"\nHyperparameter search report written to: {report_path}")


if __name__ == "__main__":
    run_hyperparameter_search()
