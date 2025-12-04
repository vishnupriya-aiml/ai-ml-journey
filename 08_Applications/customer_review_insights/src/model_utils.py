"""
model_utils.py
Utility functions to load the trained sentiment model and make predictions.
"""

import os
import joblib
import numpy as np


MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "sentiment_model.pkl")
VEC_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")


def load_model_and_vectorizer():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    if not os.path.exists(VEC_PATH):
        raise FileNotFoundError(f"Vectorizer file not found at {VEC_PATH}")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    return model, vectorizer


def predict_sentiments(texts):
    """
    texts: list of raw review strings
    returns: dict with predictions, probabilities, and label mapping
    """
    if isinstance(texts, str):
        texts = [texts]

    model, vectorizer = load_model_and_vectorizer()

    X_vec = vectorizer.transform(texts)
    probs = model.predict_proba(X_vec)
    preds = model.predict(X_vec)

    # Map: 0 -> negative, 1 -> positive
    label_map = {0: "negative", 1: "positive"}
    pred_labels = [label_map[int(p)] for p in preds]

    return {
        "pred_int": preds,
        "pred_label": pred_labels,
        "probs": probs,
        "label_map": label_map,
    }


if __name__ == "__main__":
    sample_texts = [
        "I love this app, it is very helpful!",
        "It keeps crashing, terrible experience.",
    ]
    result = predict_sentiments(sample_texts)
    print(result["pred_label"])
    print(result["probs"])
