"""
preprocessing_template.py
Reusable data preprocessing functions for ML projects.

What it does:
- Load a CSV dataset
- Split into features (X) and target (y)
- Build a preprocessing pipeline:
    - StandardScaler for numeric features
    - OneHotEncoder for categorical features
- Split into train/test sets
- Return processed arrays + the pipeline object
"""

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load CSV into a pandas DataFrame.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def split_features_target(
    df: pd.DataFrame,
    target_col: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into X (features) and y (target).
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def build_preprocessing_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    - scales numeric features
    - one-hot encodes categorical features
    """
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def preprocess_dataset(
    csv_path: str,
    target_col: str,
    numeric_features: List[str],
    categorical_features: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Full preprocessing pipeline:

    1. Load CSV
    2. Split into X, y
    3. Train/test split
    4. Fit preprocessing pipeline on X_train
    5. Transform X_train and X_test

    Returns:
        X_train_proc, X_test_proc, y_train, y_test, preprocessor
    """
    df = load_dataset(csv_path)
    X, y = split_features_target(df, target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = build_preprocessing_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    return X_train_proc, X_test_proc, y_train, y_test, preprocessor


if __name__ == "__main__":
    # Small self-test using students_sample.csv
    DATA_PATH = os.path.join("data", "students_sample.csv")
    NUMERIC_FEATURES = ["hours_studied", "attendance_rate"]
    CATEGORICAL_FEATURES = ["gender", "school_type"]

    X_train_p, X_test_p, y_train, y_test, prep = preprocess_dataset(
        DATA_PATH,
        target_col="passed",
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )

    print("X_train shape:", X_train_p.shape)
    print("X_test shape:", X_test_p.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
