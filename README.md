# Vishnu Priya – AI/ML Engineering Portfolio

Welcome to my AI/ML engineering portfolio.  
This repository is a collection of real end-to-end projects focused on:

- clean coding
- structured workflows
- practical ML/NLP problem-solving
- building models that can be deployed and used in real applications

All projects include complete pipelines — data, preprocessing, model training, evaluation, and where relevant, small apps.

---

## 🚀 Skills Overview

### **Core**
- Python (data structures, functions, file I/O)
- Data cleaning & preprocessing  
- Exploratory Data Analysis (EDA)
- Machine Learning (Scikit-Learn)
- NLP: text cleaning, TF-IDF, CountVectorizer
- Model evaluation: accuracy, precision, recall, F1, confusion matrix

### **Tools**
- VS Code  
- Jupyter Notebooks  
- Streamlit  
- Git & GitHub  
- Pandas, NumPy, Matplotlib, Seaborn  

---

## 📁 Repository Structure

AI_ML/
│
├── 00_Notes/ # Daily logs & learning documentation
├── 01_Python/ # Python foundations
├── 02_ML/
│ ├── student_performance_prediction/
│ ├── sentiment_analysis/
│ └── fake_news_detection/
│
└── README.md # (this file)


---

# ⭐ Featured ML/NLP Projects

Below are structured, industry-style ML projects with clear organization and reproducible pipelines.

---

## 1️⃣ Fake News Detection (NLP + Streamlit App)  
**Path:** `02_ML/fake_news_detection`  

A complete NLP pipeline that classifies news headlines as **REAL** or **FAKE**, with a Streamlit web application.

**Highlights**
- TF-IDF with unigrams + bigrams  
- Logistic Regression classifier  
- Evaluation: confusion matrix + classification report  
- Reusable prediction helper (`predict_text`)  
- Streamlit app (`app/app.py`) to test predictions interactively  
- Clean modular structure: preprocessing, training, evaluation, prediction modules

---

## 2️⃣ Sentiment Analysis (NLP)  
**Path:** `02_ML/sentiment_analysis`  

Binary sentiment classifier for short reviews.

**Highlights**
- Text cleaning and normalization  
- Bag-of-Words features with CountVectorizer  
- Logistic Regression model  
- Train/test split, evaluation metrics, saved model  
- Organized project layout (src, models, reports)

---

## 3️⃣ Student Performance Prediction (Tabular ML)  
**Path:** `02_ML/student_performance_prediction`  

Predicts whether a student passes based on multiple subject scores and engineered features.

**Highlights**
- Data cleaning & feature engineering  
- Logistic Regression classifier  
- EDA visualizations  
- Train/test split + evaluation  
- Saved model + reports

---

## ⚙️ How to Run Any Project

```bash
# Move into a project
cd 02_ML/fake_news_detection

# Install dependencies
pip install -r requirements.txt

# Train model
python src/train_model.py

# Evaluate model
python src/evaluate_model.py

# For projects with apps
streamlit run app/app.py

---

## 4️⃣ Image Classification with CNN (Deep Learning)

**Path:** `03_DL/image_classification_cnn`  

Convolutional Neural Network trained on the CIFAR-10 dataset (32x32 color images, 10 classes).

**Highlights**
- TensorFlow / Keras CNN architecture with multiple Conv + MaxPool blocks.
- Normalized CIFAR-10 data and structured loader (`data_loader.py`).
- Training script with callbacks (ModelCheckpoint, EarlyStopping).
- Saved best and final models (`cnn_cifar10_best.keras`, `cnn_cifar10_final.keras`).
- Separate evaluation script with classification report & confusion matrix.
- Clean structure: `data/`, `src/`, `models/`, `reports/`, `notebooks/`.
