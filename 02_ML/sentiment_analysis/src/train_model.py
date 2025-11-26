"""
train_model.py
Train a simple sentiment analysis model using Logistic Regression.
"""

from data_preprocessing import preprocess_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os


def train():
    # Preprocess + split data
    X_train, X_test, y_train, y_test, vectorizer = preprocess_data(
        "data/reviews_small.csv"
    )

    # Define model
    model = LogisticRegression()

    # Train model
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Save model and vectorizer
    joblib.dump(model, "models/sentiment_model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    # Save training results
    os.makedirs("reports", exist_ok=True)
    with open("reports/training_results.txt", "w") as f:
        f.write(f"Accuracy: {accuracy * 100:.2f}%\n")

    print("Model and vectorizer saved to models/")


if __name__ == "__main__":
    train()
