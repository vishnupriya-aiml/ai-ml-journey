"""
data_structures.py
Deep practice on Python data structures with AI/ML examples.
"""

# --------------------------------------------------
# 1) LISTS (Datasets, features, tokens)
# --------------------------------------------------

features = ["age", "income", "gender", "credit_score"]
print("Features:", features)

# Add a new feature
features.append("loan_amount")
print("Updated features:", features)

# Remove a feature
features.remove("gender")
print("Removed gender:", features)

# ML example: list of model accuracies
accuracies = [0.81, 0.76, 0.89, 0.91]

print("\nBest accuracy:", max(accuracies))
print("Average accuracy:", sum(accuracies) / len(accuracies))

# --------------------------------------------------
# 2) TUPLES (Fixed data, immutable)
# --------------------------------------------------

user_info = ("Vishnu Priya", 25, "USA")
print("\nUser info tuple:", user_info)

# Access tuple items
print("Name:", user_info[0])
print("Age:", user_info[1])

# --------------------------------------------------
# 3) SETS (Unique values, duplicates removed)
# --------------------------------------------------

labels = ["spam", "ham", "spam", "ham", "spam"]
unique_labels = set(labels)
print("\nUnique labels:", unique_labels)

# Add a new label
unique_labels.add("neutral")
print("Updated label set:", unique_labels)

# --------------------------------------------------
# 4) DICTIONARIES (Core of ML workflows)
# --------------------------------------------------

# Model parameters
model_params = {
    "learning_rate": 0.01,
    "epochs": 50,
    "optimizer": "adam"
}

print("\nModel parameters:", model_params)

# Update parameter
model_params["epochs"] = 100
print("Updated epochs:", model_params)

# Add parameter
model_params["batch_size"] = 32
print("Added batch_size:", model_params)

# ML example: predictions mapping to labels
predictions = {
    0: "spam",
    1: "ham",
    2: "spam",
    3: "neutral"
}

print("\nPredictions:")
for id, label in predictions.items():
    print(f"  Sample {id}: {label}")

# --------------------------------------------------
# 5) REAL TASK: Clean and convert a raw dataset
# --------------------------------------------------

raw_data = [
    {"name": "Alice", "score": 89, "passed": True},
    {"name": "Bob", "score": None, "passed": False},
    {"name": "Charlie", "score": 95, "passed": True}
]

# Extract scores (ignore None)
clean_scores = [d["score"] for d in raw_data if d["score"] is not None]
print("\nCleaned scores:", clean_scores)

# Compute stats
print("Max score:", max(clean_scores))
print("Average score:", sum(clean_scores) / len(clean_scores))
