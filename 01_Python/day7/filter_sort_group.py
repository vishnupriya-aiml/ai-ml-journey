"""
filter_sort_group.py
Advanced pandas operations used in real ML preprocessing:
- filtering
- sorting
- selecting columns
- grouping
- feature engineering
"""

import pandas as pd

# --------------------------------------------
# 1) Load cleaned CSV
# --------------------------------------------

df = pd.read_csv("data/students_cleaned.csv")

print("\n--- Loaded DataFrame ---")
print(df)

# --------------------------------------------
# 2) Filtering rows
# --------------------------------------------

# Students who scored above 85 in math
high_math = df[df["math"] > 85]
print("\nStudents with Math > 85:")
print(high_math)

# Students who passed
passed_students = df[df["passed"] == True]
print("\nStudents who passed:")
print(passed_students)

# --------------------------------------------
# 3) Selecting specific columns
# --------------------------------------------

subset = df[["name", "math", "english"]]
print("\nSelected columns (name, math, english):")
print(subset)

# --------------------------------------------
# 4) Sorting
# --------------------------------------------

sorted_by_math = df.sort_values(by="math", ascending=False)
print("\nSorted by math (descending):")
print(sorted_by_math)

# --------------------------------------------
# 5) Grouping & aggregation
# --------------------------------------------

grouped = df.groupby("gender").agg(
    avg_math=("math", "mean"),
    avg_science=("science", "mean"),
    avg_english=("english", "mean"),
    count=("name", "count")
)

print("\nGrouped by gender:")
print(grouped)

# --------------------------------------------
# 6) Feature Engineering
# --------------------------------------------

# Total score
df["total_score"] = df["math"] + df["science"] + df["english"]

# Average score
df["avg_score"] = df["total_score"] / 3

# High performer label
df["high_performer"] = df["avg_score"] > 85

print("\nAfter Feature Engineering:")
print(df)

# Save final version
output = "data/students_features.csv"
df.to_csv(output, index=False)
print(f"\nSaved engineered dataset to: {output}")
