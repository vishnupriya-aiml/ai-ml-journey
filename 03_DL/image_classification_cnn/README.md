# Image Classification with CNN (TensorFlow / Keras)

This project implements a Convolutional Neural Network (CNN) for image classification
using TensorFlow/Keras. The goal is to build a clean, modular deep learning pipeline
that can be trained, evaluated, and reused.

## 🎯 Objectives

- Load and preprocess image data (e.g., CIFAR-10 or similar dataset)
- Build a CNN model using TensorFlow / Keras
- Train and evaluate the model with accuracy and loss metrics
- Save the trained model for later inference
- Keep the project structure clean and maintainable

## 🗂 Project Structure

- `data/` – Raw or downloaded dataset (if stored locally)
- `models/` – Saved trained CNN models
- `notebooks/` – EDA / experiments in Jupyter
- `reports/` – Training logs, evaluation results, and structure docs
- `src/`:
  - `data_loader.py` – functions to load and preprocess image data
  - `model_builder.py` – CNN architecture definition
  - `train.py` – training loop and callbacks
  - `evaluate.py` – evaluation scripts

## 🚀 Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib (for plotting training curves)

More details will be added as the project evolves.
