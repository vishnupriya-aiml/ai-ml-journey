def encode_gender(data):
    """Convert Male/Female to numeric values."""
    for row in data:
        if row["gender"] == "Male":
            row["gender"] = 1
        else:
            row["gender"] = 0
    return data
