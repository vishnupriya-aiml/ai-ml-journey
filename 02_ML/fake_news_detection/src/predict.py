"""
predict.py
Helper functions to load the fake news model and predict on new text.
"""

import joblib
from .data_preprocessing import clean_text


def load_model_and_vectorizer(model_path="models/fake_news_model.pkl",
                              vectorizer_path="models/vectorizer.pkl"):
    """
    Load trained model and TF-IDF vectorizer from disk.
    """
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def predict_text(text: str) -> str:
    """
    Predict label (FAKE / REAL) for a single news text.
    Returns the predicted label as a string.
    """
    model, vectorizer = load_model_and_vectorizer()

    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]
    return pred
