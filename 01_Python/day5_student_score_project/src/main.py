from cleaning import load_data, fill_missing_numeric
from analysis import compute_average, compute_pass_rate
from encoding import encode_gender

def main():
    print("\n--- Student Score Analyzer ---")

    # Load data
    data = load_data("data/students.csv")

    # Fill missing numeric values (mean academic score fallback)
    data = fill_missing_numeric(data, "math", 70)
    data = fill_missing_numeric(data, "science", 70)
    data = fill_missing_numeric(data, "english", 70)

    # Encode gender
    data = encode_gender(data)

    # Compute stats
    math_avg = compute_average(data, "math")
    science_avg = compute_average(data, "science")
    english_avg = compute_average(data, "english")

    pass_rate = compute_pass_rate(data)

    print(f"Average Math Score: {math_avg:.2f}")
    print(f"Average Science Score: {science_avg:.2f}")
    print(f"Average English Score: {english_avg:.2f}")
    print(f"Pass Rate: {pass_rate:.2f}%")

if __name__ == "__main__":
    main()
