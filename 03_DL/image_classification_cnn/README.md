# Image Classification with CNN (CIFAR-10, TensorFlow/Keras)

This project implements a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset (32x32 color images across 10 classes such as airplane, car, bird, cat, dog, etc.).

The focus is on a clean, modular deep learning pipeline:
- data loading & normalization
- model definition
- training with callbacks
- evaluation with proper metrics
- saved models for reuse

---

## 🧠 Model & Data

- **Dataset:** CIFAR-10 (50,000 train images, 10,000 test images)
- **Input shape:** (32, 32, 3)
- **Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Model:** CNN with multiple Conv2D + MaxPooling2D blocks,
  followed by Dense layers and Dropout for regularization.

---

## 🗂 Project Structure

- `src/data_loader.py` – loads CIFAR-10 and normalizes pixel values to [0, 1]
- `src/model_builder.py` – defines and compiles the CNN architecture
- `src/train.py` – trains the CNN with callbacks (ModelCheckpoint, EarlyStopping)
- `src/evaluate.py` – loads the best saved model and generates metrics
- `models/` – contains saved models (`cnn_cifar10_best.keras`, `cnn_cifar10_final.keras`)
- `reports/` – training log and evaluation report (classification report + confusion matrix)
- `notebooks/` – reserved for future EDA/experiment notebooks

---

## ⚙️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- scikit-learn (for evaluation metrics)
- Matplotlib (for future plots)

---

## 🚀 How to Run

From the project root:

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python src/train.py

# Evaluate the saved best model
python src/evaluate.py
