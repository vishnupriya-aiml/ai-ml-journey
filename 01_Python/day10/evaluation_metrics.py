"""
evaluation_metrics.py
Evaluate a Logistic Regression model using:
- confusion matrix
- precision
- recall
- F1-score
- accuracy
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

# -------------------------------------------
# 1) Load engineered dataset
# -------------------------------------------

df = pd.read_csv("data/students_features.csv")
print("\nLoaded Data:")
print(df.head())

# -------------------------------------------
# 2) Define features and target
# -------------------------------------------

X = df[["math", "science", "english", "gender_encoded", "total_score", "avg_score"]]
y = df["passed"]

# -------------------------------------------
# 3) Train/Test split
# -------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# -------------------------------------------
# 4) Train Logistic Regression
# -------------------------------------------

model = LogisticRegression()
model.fit(X_train, y_train)

# -------------------------------------------
# 5) Predictions
# -------------------------------------------

y_pred = model.predict(X_test)

# -------------------------------------------
# 6) Evaluation Metrics
# -------------------------------------------

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nAccuracy: {:.2f}%".format(acc * 100))
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
