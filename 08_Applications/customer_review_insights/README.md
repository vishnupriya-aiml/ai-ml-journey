# Customer Review Insights Dashboard

An internal-style analytics dashboard for product and support teams to
analyze customer reviews using an ML-based sentiment classifier.

## 🔍 Overview

This app:

- Ingests customer reviews from a CSV file
- Uses a trained ML model (TF-IDF + Logistic Regression) to predict sentiment
  (positive / negative)
- Computes summary metrics:
  - total reviews
  - positive vs negative percentages
  - sentiment breakdown by source (e.g., App Store, Play Store, Web, Email)
- Surfaces high-confidence negative reviews for support follow-up
- Displays raw data with predictions for ad-hoc analysis

## 🧠 ML Model

- Features: TF-IDF on `review_text` with unigrams and bigrams
- Model: Logistic Regression
- Training script: `src/train_model.py`
- Artifacts:
  - `models/sentiment_model.pkl`
  - `models/tfidf_vectorizer.pkl`
- Training report:
  - `reports/training_report.txt`

## 🗂 Project Structure

```text
customer_review_insights/
├── app/
│   └── app.py                # Streamlit dashboard
├── data/
│   └── reviews_sample.csv    # Example reviews dataset
├── models/
│   ├── sentiment_model.pkl   # Trained model
│   └── tfidf_vectorizer.pkl  # Fitted TF-IDF vectorizer
├── reports/
│   └── training_report.txt   # Model evaluation summary
├── src/
│   ├── __init__.py
│   ├── data_utils.py         # Load/clean datasets, attach predictions
│   ├── model_utils.py        # Load model + vectorizer, prediction helpers
│   ├── analytics.py          # Summary stats and sentiment by source
│   └── train_model.py        # Training script
├── README.md
└── requirements.txt
