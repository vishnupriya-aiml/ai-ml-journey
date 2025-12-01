"""
train_model.py
Train a fake news detection model using TF-IDF features + Logistic Regression.
"""

from data_preprocessing import preprocess_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os


def train():
    # Preprocess data and get train/test splits
    X_train, X_test, y_train, y_test, vectorizer = preprocess_data(
        "data/fake_news_small.csv"
    )

    # Define model
    model = LogisticRegression(max_iter=1000)

    # Train model
    model.fit(X_train, y_train)

    # Evaluate on test split
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Ensure directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # Save model + vectorizer
    joblib.dump(model, "models/fake_news_model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    # Save simple training results
    with open("reports/training_results.txt", "w") as f:
        f.write(f"Accuracy: {accuracy * 100:.2f}%\n")

    print("Model and vectorizer saved in models/ folder.")
    print("Training results written to reports/training_results.txt")


if __name__ == "__main__":
    train()
