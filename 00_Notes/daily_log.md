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

---

### Fake News Detection – Data & Preprocessing (Dec 01, 2025)

**Learned / Did:**
- Created a small labeled fake vs real news dataset.
- Implemented text cleaning for news headlines.
- Used TF-IDF (unigrams + bigrams, stopwords removed) to convert text to features.
- Performed train/test split with stratification.

**Built:**
- `02_ML/fake_news_detection/data/fake_news_small.csv`
- `02_ML/fake_news_detection/src/data_preprocessing.py`

---

### Fake News Detection – Model Training (Dec 1, 2025)

**Learned / Did:**
- Trained Logistic Regression classifier on TF-IDF features for fake vs real news.
- Saved model and TF-IDF vectorizer to disk.
- Evaluated model with confusion matrix and classification report.
- Built a reusable prediction helper (`predict_text`) for later use in a Streamlit app.

**Built:**
- `src/train_model.py`
- `src/evaluate_model.py`
- `src/predict.py`
- `models/fake_news_model.pkl`
- `models/vectorizer.pkl`
- `reports/training_results.txt`
- `reports/evaluation_report.txt`

---

### Fake News Detection – Streamlit App (Dec 01, 2025)

**Learned / Did:**
- Built a Streamlit UI for fake news detection.
- Integrated existing model + prediction helper into a web app.
- Tested predictions on different news-like texts.

**Built:**
- `02_ML/fake_news_detection/app/app.py`


---

### Portfolio – GitHub Top-Level README (Dec 01, 2025)

**Learned / Did:**
- Wrote a professional top-level README for the AI_ML repo.
- Documented key projects and repository structure for recruiters.
- Turned the repo into a portfolio entry point.

**Built:**
- `AI_ML/README.md`

---

### Deep Learning – CNN Project Skeleton (Dec 01, 2025)

**Learned / Did:**
- Created a structured project for image classification using CNNs.
- Prepared folders for data, models, notebooks, src, and reports.
- Added basic README and requirements for TensorFlow/Keras.

**Built:**
- `03_DL/image_classification_cnn` project skeleton

---

### Deep Learning – CIFAR-10 Data Loader (Dec 1, 2025)

**Learned / Did:**
- Installed TensorFlow (Keras) for deep learning.
- Implemented a data loader for the CIFAR-10 image dataset.
- Normalized images to [0, 1] and confirmed train/test shapes.

**Built:**
- `03_DL/image_classification_cnn/src/data_loader.py`

---

### Deep Learning – CNN Training (Dec 1, 2025)

**Learned / Did:**
- Implemented a training script for a CNN on the CIFAR-10 dataset.
- Used callbacks (ModelCheckpoint, EarlyStopping) to manage training.
- Evaluated the model on the test set and saved the best and final models.
- Logged training details to a report file.

**Built:**
- `03_DL/image_classification_cnn/src/train.py`
- `03_DL/image_classification_cnn/models/cnn_cifar10_best.keras`
- `03_DL/image_classification_cnn/models/cnn_cifar10_final.keras`
- `03_DL/image_classification_cnn/reports/training_log.txt`


---

### Deep Learning – CNN Evaluation (Dec 1, 2025)

**Learned / Did:**
- Loaded the saved CNN model and evaluated it on the CIFAR-10 test set.
- Generated a full classification report and confusion matrix.
- Saved evaluation metrics into a report file for documentation.

**Built:**
- `03_DL/image_classification_cnn/src/evaluate.py`
- `03_DL/image_classification_cnn/reports/evaluation_report.txt`
- Updated `03_DL/image_classification_cnn/requirements.txt`

---

### Deep Learning – CNN Project Documentation & Portfolio Integration (Dec 1, 2025)

**Learned / Did:**
- Added the CNN image classification project to the main AI_ML portfolio README.
- Documented the CNN project with a focused README including structure and run instructions.
- Prepared LinkedIn and resume descriptions for the CNN project.

**Built/Updated:**
- `AI_ML/README.md`
- `03_DL/image_classification_cnn/README.md`
- LinkedIn + resume content for CNN project


---

### Deep Learning NLP – LSTM Dataset & Loader (Dec 1, 2025)

**Learned / Did:**
- Created a custom sentiment dataset for LSTM.
- Implemented a clean data loader to read text and labels.

**Built:**
- `04_NLP/sentiment_lstm/data/reviews_lstm.csv`
- `04_NLP/sentiment_lstm/src/data_loader.py`

---

### Deep Learning NLP – LSTM Tokenization & Padding (Dec 1, 2025)

**Learned / Did:**
- Implemented tokenization and padded sequence generation for sentiment texts.
- Encoded text labels into numeric classes (0 = negative, 1 = positive).
- Created a reusable preprocessing pipeline that splits data into train/test.

**Built:**
- `04_NLP/sentiment_lstm/src/text_preprocessor.py`

---

### Deep Learning NLP – LSTM Model Builder (Dec 1, 2025)

**Learned / Did:**
- Defined an LSTM-based model for binary sentiment classification.
- Used Embedding + LSTM + Dense layers with dropout for regularization.
- Compiled the model with Adam and binary crossentropy and verified the architecture via `model.summary()`.

**Built:**
- `04_NLP/sentiment_lstm/src/model_builder.py`

---

### Deep Learning NLP – LSTM Training (Dec 1, 2025)

**Learned / Did:**
- Connected data loading, preprocessing, and model builder into a full LSTM training pipeline.
- Trained an LSTM-based sentiment classifier with validation split and callbacks.
- Saved best and final models and wrote training logs to the reports folder.

**Built:**
- `04_NLP/sentiment_lstm/src/train.py`
- `04_NLP/sentiment_lstm/models/lstm_sentiment_best.keras`
- `04_NLP/sentiment_lstm/models/lstm_sentiment_final.keras`
- `04_NLP/sentiment_lstm/reports/training_log.txt`

---

### Deep Learning NLP – LSTM Evaluation (Dec 1, 2025)

**Learned / Did:**
- Loaded the saved LSTM sentiment model and evaluated it on the test set.
- Generated a classification report and confusion matrix for positive/negative sentiment.
- Saved evaluation metrics into an evaluation report file.

**Built:**
- `04_NLP/sentiment_lstm/src/evaluate.py`
- `04_NLP/sentiment_lstm/reports/evaluation_report.txt`


---

### Transformer NLP – BERT Project Setup (Dec 2, 2025)

**Created:**
- transformer_bert project folder and full structure
- requirements with TensorFlow + HuggingFace
- dataset (sentiment_bert.csv)
- placeholder README
- data_loader module

**Status:**
BERT project structure ready. Next step: tokenizer & encoding pipeline.


---

### Transformer NLP – BERT Tokenization & Encoding (Dec 2, 2025)

**Learned / Did:**
- Loaded a pre-trained BERT tokenizer via HuggingFace.
- Implemented encoding for texts into input_ids and attention_mask.
- Encoded sentiment labels into numeric classes and split train/test sets.

**Built:**
- `04_NLP/transformer_bert/src/tokenizer_builder.py`

---

### Transformer NLP – BERT Model Builder (Dec 2, 2025)

**Learned / Did:**
- Loaded a pre-trained BERT model (`bert-base-uncased`) with a sequence classification head.
- Wrapped it as a TensorFlow/Keras model for binary sentiment classification.
- Compiled the model with Adam optimizer and the built-in classification loss.

**Built:**
- `04_NLP/transformer_bert/src/model_builder.py`

---

### Transformer NLP – Training (Dec 2, 2025)

**Learned / Did:**
- Connected dataset, tokenizer, and Transformer classifier into a full training pipeline.
- Trained a Transformer-style sentiment classifier using token IDs from a BERT tokenizer.
- Saved best and final model weights and wrote a training log.

**Built:**
- `04_NLP/transformer_bert/src/train.py`
- `04_NLP/transformer_bert/models/transformer_sentiment_best.keras`
- `04_NLP/transformer_bert/models/transformer_sentiment_final.keras`
- `04_NLP/transformer_bert/reports/training_log.txt`

---

### Transformer NLP – Evaluation (Dec 2, 2025)

**Learned / Did:**
- Loaded the trained Transformer-style sentiment classifier and evaluated it on the test set.
- Generated a classification report and confusion matrix for negative/positive sentiment.
- Saved evaluation metrics into an evaluation report file.

**Built:**
- `04_NLP/transformer_bert/src/evaluate.py`
- `04_NLP/transformer_bert/reports/evaluation_report.txt`

### GRU Text Classifier (Dec 3, 2025)
- Created full dataset, tokenizer, GRU model, training and evaluation pipeline.
- Saved best and final model.
- Generated evaluation report.
- Project ready for portfolio.

---

### BiLSTM Text Classifier (Dec 3, 2025)

**Learned / Did:**
- Implemented a Bidirectional LSTM model for sentiment classification.
- Built tokenization, padding, training, and evaluation pipeline similar to GRU.
- Saved best and final BiLSTM models and generated evaluation metrics.

**Built:**
- `03_DL/bilstm_text_classifier/data/sentiment_bilstm.csv`
- `03_DL/bilstm_text_classifier/src/data_loader.py`
- `03_DL/bilstm_text_classifier/src/tokenizer_builder.py`
- `03_DL/bilstm_text_classifier/src/model_builder.py`
- `03_DL/bilstm_text_classifier/src/train.py`
- `03_DL/bilstm_text_classifier/src/evaluate.py`
- `03_DL/bilstm_text_classifier/models/*`
- `03_DL/bilstm_text_classifier/reports/*`


---

### Multi-class CNN Image Classifier (Dec 3, 2025)

**Learned / Did:**
- Used CIFAR-10 and filtered it to three classes (cat, dog, frog).
- Implemented a multi-class CNN with convolution, pooling, and dense layers.
- Trained the model with validation split, checkpointing, and early stopping.
- Evaluated using accuracy, classification report, and confusion matrix.

**Built:**
- `03_DL/cnn_multiclass_classifier/src/data_loader.py`
- `03_DL/cnn_multiclass_classifier/src/model_builder.py`
- `03_DL/cnn_multiclass_classifier/src/train.py`
- `03_DL/cnn_multiclass_classifier/src/evaluate.py`
- `03_DL/cnn_multiclass_classifier/models/*`
- `03_DL/cnn_multiclass_classifier/reports/*`
