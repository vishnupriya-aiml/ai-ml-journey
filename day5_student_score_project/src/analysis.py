def compute_average(data, column):
    """Compute average score for a numeric column."""
    values = [float(row[column]) for row in data]
    return sum(values) / len(values)

def compute_pass_rate(data):
    """Percentage of students who passed."""
    total = len(data)
    passed = sum(1 for row in data if row["passed"] == "True")
    return (passed / total) * 100
