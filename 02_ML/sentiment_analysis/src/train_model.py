"""
train_model.py
Temporary test to verify preprocessing works.
"""

from data_preprocessing import preprocess_data

def main():
    X_train, X_test, y_train, y_test, vectorizer = preprocess_data(
        "data/reviews_small.csv"
    )

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Sample train labels:", list(y_train)[:5])

if __name__ == "__main__":
    main()
"""
train_model.py
Temporary test to verify preprocessing works.
"""

from data_preprocessing import preprocess_data

def main():
    X_train, X_test, y_train, y_test, vectorizer = preprocess_data(
        "data/reviews_small.csv"
    )

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Sample train labels:", list(y_train)[:5])

if __name__ == "__main__":
    main()
