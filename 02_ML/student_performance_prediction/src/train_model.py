import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from data_preprocessing import load_data, prepare_features

def train():
    df = load_data("data/students_features.csv")
    X, y = prepare_features(df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predictions + accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save model
    joblib.dump(model, "models/model.pkl")

    # Save training accuracy
    with open("reports/training_results.txt", "w") as f:
        f.write(f"Accuracy: {accuracy * 100:.2f}%\n")

    print("Training completed.")
    print(f"Model accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    train()
