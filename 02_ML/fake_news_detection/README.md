# Fake News Detection – NLP Machine Learning Project

This project builds a Machine Learning/NLP pipeline to classify news headlines/articles as **FAKE** or **REAL**.

## 🎯 Project Goals

- Load and clean real-world text data (fake vs real news).
- Convert raw text into numeric features (e.g., TF-IDF).
- Train a classification model (Logistic Regression / Linear SVM).
- Evaluate the model with practical metrics (accuracy, precision, recall, F1, confusion matrix).
- Expose the model via a simple **Streamlit web app** where users can enter a headline and see the prediction.

## 🗂 Project Structure

- `data/` – raw and processed datasets (fake vs real news).
- `models/` – saved trained model and vectorizer.
- `notebooks/` – EDA and experimentation notebooks.
- `src/` – core Python modules:
  - `data_preprocessing.py` – load and clean text, vectorization.
  - `train_model.py` – training pipeline.
  - `evaluate_model.py` – metrics and evaluation.
  - `predict.py` – helper for loading model and predicting single headlines.
- `app/` – Streamlit application entrypoint (`app.py`).
- `reports/` – training results, evaluation summary, project structure.
- `requirements.txt` – Python dependencies for the project.

More details and metrics will be added as the project is developed.
