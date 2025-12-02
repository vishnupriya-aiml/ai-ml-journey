# Sentiment Classification using LSTM (TensorFlow/Keras)

This project builds a deep learning model using LSTM layers to classify text
into positive or negative sentiment. The project follows a clean ML engineering
pipeline with separate modules for data loading, preprocessing, model building,
training, and evaluation.

## 🎯 Goals
- Tokenize text and build padded sequences
- Create an LSTM model using TensorFlow/Keras
- Train on sentiment dataset (we will create a custom dataset)
- Evaluate using accuracy, classification report, and confusion matrix
- Save trained model for reuse

## 🗂 Structure
- `src/data_loader.py` – load raw dataset
- `src/text_preprocessor.py` – tokenize & pad sequences
- `src/model_builder.py` – LSTM model architecture
- `src/train.py` – training loop with callbacks
- `src/evaluate.py` – evaluation scripts
- `data/` – dataset (CSV)
- `models/` – saved LSTM models
- `reports/` – logs, metrics, evaluation report

## ⚙️ Tech Stack
- TensorFlow/Keras
- Pandas
- NumPy
- scikit-learn
- Matplotlib
