# Daily Engineering Log

---

### 📅 Date: Nov 16, 2025

**Learned:**  
- Control flow (if/elif/else)  
- For loops  
- While loops  
- Functions (parameters, return values)  
- Multiple return values  

**Built:**  
- `control_flow.py`

**Confusion / Questions:**  
- (Write anything here. If nothing, leave blank.)

**Tomorrow’s Plan:**  
- Deep practice on Python data structures  
  - lists  
  - tuples  
  - sets  
  - dictionaries  
- More clean code practice

---


### Worked On (Continuation of Nov 17, 2025)

**Learned:**  
- Lists (append, remove, stats)  
- Tuples (immutable)  
- Sets (unique labels)  
- Dictionaries (model params, predictions)  
- Comprehensions (cleaning raw dataset)

**Built:**  
- data_structures.py

**Confusion:**  
- (write any if confused, otherwise leave blank)

**Next:**  
- File handling  
- Working with CSV/JSON  
- Functions for preprocessing  


### Continued Work (Nov 17, 2025)

**Learned:**  
- File handling (txt, csv, json)  
- Writing + reading data files  
- Using CSV module without Pandas  
- JSON serialization/deserialization  
- Extracting values for ML parameters  

**Built:**  
- file_handling.py  
- sample CSV, JSON, text files  


### Mini Project Completed (Nov 17, 2025)

**Project:** Student Score Analyzer  
**Skills practiced:**  
- CSV loading & cleaning  
- Missing value handling  
- Encoding categorical values  
- Modular Python  
- Pipeline-style ML preprocessing  

**Files built:**  
- students.csv  
- cleaning.py  
- analysis.py  
- encoding.py  
- main.py 

---

### 📅 Date: Nov 18, 2025

**Learned:**  
- Installing and importing pandas  
- Loading CSV into DataFrame  
- Viewing head, shape, columns  
- describe(), isna(), basic EDA  
- Filling missing numeric values  
- Encoding categorical column with map()  
- Saving cleaned CSV files

**Built:**  
- `day6/pandas_basics.py`  
- `students_cleaned.csv`

**Confusion / Questions:**  
- (Write if anything felt unclear; if none, leave blank)

**Next Plan:**  
- More pandas practice: filtering, sorting, grouping  
- Start thinking in “DataFrame mindset” like an ML engineer


---

### 📅 Date: Nov 19, 2025

**Learned:**  
- Pandas filtering with conditions  
- Column selection for ML pipelines  
- Sorting DataFrames  
- Groupby + aggregations (mean, count)  
- Feature engineering:
  - total_score  
  - avg_score  
  - high_performer label  
- Saving engineered datasets

**Built:**  
- `day7/filter_sort_group.py`  
- `students_features.csv`

**Confusion / Notes:**  
- (write anything here)

**Next Plan:**  
- Pandas: merging, joining, concatenating  
- Prepare for first REAL ML model (Logistic Regression)

---

### Continued Work — Nov 19, 2025

**Learned:**  
- merge() for combining datasets  
- join() with DataFrame indexes  
- concat() for stacking data  
- inner joins in ML  
- indexing and key-based merging  
- preparing multi-source datasets for modeling  

**Built:**  
- merge_join_concat.py  
- students_merged.csv  

**Next:**  
- COMMAND 4: First ML model (Logistic Regression on cleaned dataset)



---

### ML Work — Nov 21, 2025

**Learned:**  
- First ML pipeline (Logistic Regression)  
- Feature selection  
- Train-test split  
- Model training  
- Model evaluation (accuracy score)  
- Saving model predictions  

**Built:**  
- logistic_regression_model.py  
- model_predictions.csv  

**Next Plan:**  
- COMMAND 4 PART 2: confusion matrix, metrics, ROC curve  
- COMMAND 5: Build your FIRST end-to-end ML project folder



---

### 📅 Date: Nov 22, 2025

**Learned:**  
- Train/test split basics for evaluation  
- Confusion matrix  
- Precision, recall, F1-score  
- How classification_report summarizes model quality  

**Built:**  
- `day10/evaluation_metrics.py`

**Notes / Confusion:**  
- (write if any metrics feel confusing; we can clarify them in the next step)

**Next Plan:**  
- Turn this into a proper mini ML project folder  
- Begin designing your first portfolio-ready ML project

---

### Continued – Nov 22, 2025

**Learned / Did:**
- Created EDA notebook with:
  - basic DataFrame inspection
  - missing value check
  - target distribution
  - score histograms
  - boxplots & barplots
- Documented project structure and reports.
- Finalized student performance prediction as a portfolio-ready ML project.

**Built:**
- `notebooks/eda.ipynb`
- Updated `reports/README.md`
- Updated `reports/project_structure.md`

---

### Continued – Nov 25, 2025

**Learned / Did:**
- Designed a clean ML project structure for an NLP/Sentiment Analysis project.
- Created folders for data, models, notebooks, src, and reports.
- Added root README and basic documentation.

**Built:**
- `ML/sentiment_analysis` project skeleton


---

### Sentiment Project – (date you want, e.g., Nov 25, 2025)

**Learned / Did:**
- Designed a small labeled text dataset for sentiment analysis.
- Implemented text cleaning (lowercase, remove punctuation, normalize spaces).
- Built preprocessing pipeline using:
  - CountVectorizer
  - train/test split
- Tested preprocessing through a small script.

**Built:**
- `ML/sentiment_analysis/data/reviews_small.csv`
- `ML/sentiment_analysis/src/data_preprocessing.py`
- test logic in `train_model.py`

---

### Sentiment Project – Training Complete (Nov 25, 2025)

**Learned / Did:**
- Used CountVectorizer to convert text to features.
- Trained Logistic Regression model for sentiment classification.
- Saved model and vectorizer to `models/`.
- Evaluated model using confusion matrix and classification report.

**Built:**
- `sentiment_model.pkl`
- `vectorizer.pkl`
- `src/train_model.py`
- `src/evaluate_model.py`
- `reports/training_results.txt`

---

### Fake News Project – Skeleton Created (December 1, 2025)

**Learned / Did:**
- Designed full project structure for a Fake News Detection ML/NLP system.
- Prepared folders for data, models, notebooks, src, reports, and app.
- Wrote initial project README and dependencies list.

**Built:**
- `02_ML/fake_news_detection` project skeleton
