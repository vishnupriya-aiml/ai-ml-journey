"""
merge_join_concat.py
Advanced pandas operations:
- merge()
- join()
- concat()

Used heavily in ML data preparation.
"""

import pandas as pd

# -----------------------------------------
# 1) Load datasets
# -----------------------------------------

df_basic = pd.read_csv("data/students_basic.csv")
df_scores = pd.read_csv("data/student_scores.csv")

print("\n--- Basic Info Data ---")
print(df_basic)

print("\n--- Scores Data ---")
print(df_scores)

# -----------------------------------------
# 2) MERGE (most common in ML pipelines)
# -----------------------------------------

merged = pd.merge(df_basic, df_scores, on="student_id", how="inner")

print("\n--- Merged DataFrame ---")
print(merged)

# -----------------------------------------
# 3) JOIN (when index is key)
# -----------------------------------------

df_basic_indexed = df_basic.set_index("student_id")
df_scores_indexed = df_scores.set_index("student_id")

joined = df_basic_indexed.join(df_scores_indexed)

print("\n--- Joined DataFrame ---")
print(joined)

# -----------------------------------------
# 4) CONCAT (stacking datasets)
# -----------------------------------------

# Example: splitting dataset then concatenating
top_students = merged.head(2)
remaining_students = merged.tail(3)

combined = pd.concat([top_students, remaining_students], axis=0)

print("\n--- Concatenated DataFrame ---")
print(combined)

# -----------------------------------------
# 5) Save final merged dataset
# -----------------------------------------

output_path = "data/students_merged.csv"
merged.to_csv(output_path, index=False)

print(f"\nSaved merged dataset to: {output_path}")
