import csv

def load_data(path):
    """Load CSV into a list of dictionaries."""
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def fill_missing_numeric(data, column, fill_value):
    """Fill missing numeric values with a given fill value."""
    for row in data:
        if row[column] == "" or row[column] is None:
            row[column] = fill_value
        else:
            row[column] = float(row[column])
    return data
