"""
logistic_regression_model.py
First ML model pipeline:
- load data
- preprocess
- train Logistic Regression
- evaluate accuracy
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------------------------------
# 1) Load engineered dataset
# -------------------------------------------

df = pd.read_csv("data/students_features.csv")
print("\nLoaded Data:")
print(df.head())

# -------------------------------------------
# 2) Features and target
# -------------------------------------------

X = df[["math", "science", "english", "gender_encoded", "total_score", "avg_score"]]
y = df["passed"]

# -------------------------------------------
# 3) Train/Test Split
# -------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------------------------------
# 4) Train Logistic Regression
# -------------------------------------------

model = LogisticRegression()
model.fit(X_train, y_train)

# -------------------------------------------
# 5) Evaluate model
# -------------------------------------------

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# -------------------------------------------
# 6) Save predictions for later use
# -------------------------------------------

pred_df = X_test.copy()
pred_df["actual_passed"] = y_test.values
pred_df["predicted_passed"] = y_pred

pred_df.to_csv("data/model_predictions.csv", index=False)

print("\nSaved predictions to data/model_predictions.csv")
