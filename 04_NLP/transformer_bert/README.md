# Transformer-based Sentiment Classification (BERT Tokenizer + Keras Transformer)

This project implements a modern NLP sentiment classifier that combines:

- A **BERT-style tokenizer** (HuggingFace) to convert text into token IDs  
- A custom **Transformer encoder** built with TensorFlow/Keras 
  (MultiHeadAttention + positional embeddings)  
- A classification head for **positive / negative** sentiment

The focus is on a clean, modular engineering pipeline similar to real-world
AI/ML projects.

---

## 🧠 Overview

- Input: short text reviews
- Output: sentiment label (negative / positive)
- Tokenization: BERT-style WordPiece tokenizer
- Model: Keras Transformer encoder with multi-head attention
- Training: Adam optimizer, early stopping, model checkpointing
- Evaluation: accuracy, classification report, confusion matrix

---

## 🗂 Project Structure

```text
transformer_bert
├── data
│   └── sentiment_bert.csv
├── models
│   ├── transformer_sentiment_best.keras
│   └── transformer_sentiment_final.keras
├── notebooks
├── reports
│   ├── training_log.txt
│   └── evaluation_report.txt
└── src
    ├── __init__.py
    ├── data_loader.py
    ├── tokenizer_builder.py
    ├── model_builder.py
    ├── train.py
    └── evaluate.py
