# Project Structure

```text
student_performance_prediction
│
├── data/
│   └── students_features.csv          # Engineered dataset used for modeling
│
├── models/
│   └── model.pkl                      # Trained Logistic Regression model
│
├── notebooks/
│   └── eda.ipynb                      # Exploratory data analysis notebook
│
├── reports/
│   ├── training_results.txt           # Accuracy and key metrics from training
│   ├── project_structure.md           # This file – structure overview
│   └── README.md                      # Reports summary
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py          # Data loading and feature selection
│   ├── train_model.py                 # Training pipeline script
│   └── evaluate_model.py              # Evaluation script (confusion matrix, report)
│
├── requirements.txt                   # Python dependencies
└── README.md                          # Top-level project description
