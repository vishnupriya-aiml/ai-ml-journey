"""
pandas_basics.py
First steps with pandas DataFrame for ML-style data analysis.
"""

import pandas as pd

# ----------------------------------------
# 1) Load CSV into a DataFrame
# ----------------------------------------

df = pd.read_csv("data/students.csv")

print("DataFrame loaded.\n")
print("Head:")
print(df.head())

print("\nShape (rows, columns):", df.shape)
print("\nColumns:", df.columns.tolist())

# ----------------------------------------
# 2) Basic statistics
# ----------------------------------------

print("\nBasic statistics (numeric columns):")
print(df.describe())

# ----------------------------------------
# 3) Check missing values
# ----------------------------------------

print("\nMissing values per column:")
print(df.isna().sum())

# ----------------------------------------
# 4) Fill missing numeric values with column means
# ----------------------------------------

numeric_cols = ["math", "science", "english"]

for col in numeric_cols:
    mean_value = df[col].mean()
    df[col].fillna(mean_value, inplace=True)
    print(f"Filled missing values in {col} with mean = {mean_value:.2f}")

print("\nData after filling missing values:")
print(df)

# ----------------------------------------
# 5) Encode gender (Female = 0, Male = 1)
# ----------------------------------------

df["gender_encoded"] = df["gender"].map({"Female": 0, "Male": 1})

print("\nData with encoded gender:")
print(df[["name", "gender", "gender_encoded"]])

# ----------------------------------------
# 6) Save cleaned data
# ----------------------------------------

output_path = "data/students_cleaned.csv"
df.to_csv(output_path, index=False)
print(f"\nCleaned data saved to: {output_path}")
