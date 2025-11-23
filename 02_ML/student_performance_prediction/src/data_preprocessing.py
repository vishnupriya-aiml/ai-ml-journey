import pandas as pd

def load_data(path):
    """Load the dataset."""
    df = pd.read_csv(path)
    return df

def prepare_features(df):
    """Select the ML features and target."""
    X = df[["math", "science", "english", "gender_encoded", "total_score", "avg_score"]]
    y = df["passed"]
    return X, y
