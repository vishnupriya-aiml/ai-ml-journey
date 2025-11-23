import joblib
from sklearn.metrics import confusion_matrix, classification_report
from data_preprocessing import load_data, prepare_features

def evaluate():
    model = joblib.load("models/model.pkl")

    df = load_data("data/students_features.csv")
    X, y = prepare_features(df)

    y_pred = model.predict(X)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))

    print("\nClassification Report:")
    print(classification_report(y, y_pred))

if __name__ == "__main__":
    evaluate()
