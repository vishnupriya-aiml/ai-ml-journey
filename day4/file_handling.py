"""
file_handling.py
Python file handling for ML pipelines.
"""

import json
import csv

# ------------------------------------------
# 1) Write to a text file
# ------------------------------------------

with open("notes.txt", "w") as f:
    f.write("This is a test note.\n")
    f.write("Data processed successfully.\n")

print("Text file written: notes.txt")

# ------------------------------------------
# 2) Read from a text file
# ------------------------------------------

with open("notes.txt", "r") as f:
    content = f.read()

print("\nContents of notes.txt:")
print(content)

# ------------------------------------------
# 3) Read a CSV file without pandas
# ------------------------------------------

# Create a sample CSV
sample_csv = [
    ["name", "score"],
    ["Alice", "89"],
    ["Bob", "76"],
    ["Charlie", "95"]
]

with open("scores.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(sample_csv)

print("CSV file written: scores.csv")

# Read CSV
print("\nReading scores.csv:")
with open("scores.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)

# ------------------------------------------
# 4) JSON file handling
# ------------------------------------------

data = {
    "model": "RandomForest",
    "accuracy": 0.91,
    "params": {
        "n_estimators": 200,
        "max_depth": 10
    }
}

# Write JSON
with open("model_info.json", "w") as f:
    json.dump(data, f, indent=4)

print("\nJSON file written: model_info.json")

# Read JSON
with open("model_info.json", "r") as f:
    model_data = json.load(f)

print("\nModel data loaded from JSON:")
print(model_data)

# ------------------------------------------
# 5) Real ML-style example:
# Extract accuracy and parameters
# ------------------------------------------

accuracy = model_data["accuracy"]
n_estimators = model_data["params"]["n_estimators"]

print(f"\nExtracted accuracy: {accuracy}")
print(f"Number of estimators: {n_estimators}")
