# AI / ML Journey – Vishnu Priya

This repository contains my structured journey from zero to AI/ML Engineer.  
All code, projects, and notes are organized by topic and timeline.

## 👩‍💻 About Me (Short)

- International student in the USA (MS in Technology Management).
- Actively preparing for AI/ML Engineer roles.
- Focusing on **Python, ML, NLP, and practical end-to-end projects**.
- All work here is hands-on – scripts, models, EDA, and small apps.

---

## 🗂 Repository Structure

- `00_Notes/` – Daily learning logs and planning.
- `01_Python/` – Core Python foundations (data types, control flow, files, pandas).
- `02_ML/` – Machine Learning & NLP projects:
  - `student_performance_prediction/`
  - `sentiment_analysis/`
  - `fake_news_detection/`
- Future:
  - `DL/` – Deep Learning projects
  - `NLP/` – Advanced NLP projects
  - `portfolio/`, `resume/`, `linkedin/`, `applications/`, `interview_prep/`

---

## 🔑 Key AI/ML Projects

### 1️⃣ Student Performance Prediction (Tabular ML)

**Path:** `02_ML/student_performance_prediction`  

**Description:**  
Predicts whether a student passes based on scores and engineered features.

**Highlights:**
- Clean project structure (data, src, models, reports, notebooks).
- Feature engineering (`total_score`, `avg_score`, etc.).
- Logistic Regression model with train/test split.
- Evaluation: accuracy + classification report.
- Saved model for reuse.

---

### 2️⃣ Sentiment Analysis (Text Classification)

**Path:** `02_ML/sentiment_analysis`  

**Description:**  
Classifies short text reviews as **positive** or **negative**.

**Highlights:**
- Custom text cleaning and preprocessing.
- Bag-of-words features with `CountVectorizer`.
- Logistic Regression classifier.
- Evaluation metrics and saved model + vectorizer.

---

### 3️⃣ Fake News Detection (NLP + Streamlit App)

**Path:** `02_ML/fake_news_detection`  

**Description:**  
Detects whether a news headline/article is **FAKE** or **REAL**, and exposes the model via a **Streamlit app**.

**Highlights:**
- Text cleaning and TF-IDF (`TfidfVectorizer` with unigrams + bigrams).
- Logistic Regression classifier with evaluation (confusion matrix + report).
- Reusable `predict_text()` helper in `src/predict.py`.
- Streamlit app in `app/app.py` where users can:
  - Enter a headline
  - Get a REAL/FAKE prediction
- Models and vectorizers saved in `models/`.

---

## ⚙️ How to Run (High Level)

Requirements are listed per project in their `requirements.txt`.

General steps (example):

```bash
# In project folder, e.g. fake_news_detection
pip install -r requirements.txt

# Train model
python src/train_model.py

# Evaluate model
python src/evaluate_model.py

# Run Streamlit app (for fake_news_detection)
streamlit run app/app.py
