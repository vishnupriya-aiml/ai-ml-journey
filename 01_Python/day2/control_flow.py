"""
control_flow.py
Python control flow, loops, and functions practice.
Engineer-style clean code.
"""

# ----------------------------
# 1) If / elif / else
# ----------------------------

score = 87

if score >= 90:
    print("Grade: A")
elif score >= 80:
    print("Grade: B")
elif score >= 70:
    print("Grade: C")
else:
    print("Grade: D or below")

# ----------------------------
# 2) For loop – iterate through list
# ----------------------------

fruits = ["apple", "banana", "orange"]

print("\nFruit list:")
for f in fruits:
    print(" -", f)

# ----------------------------
# 3) While loop
# ----------------------------

count = 3
print("\nCountdown:")
while count > 0:
    print(count)
    count -= 1

print("Go!")

# ----------------------------
# 4) Functions with parameters & return values
# ----------------------------

def calculate_average(numbers):
    """
    Accepts a list of numbers and returns the average.
    """
    if len(numbers) == 0:
        return None
    return sum(numbers) / len(numbers)

nums = [12, 45, 67, 23, 89]
avg = calculate_average(nums)
print("\nNumbers:", nums)
print("Average =", avg)

# ----------------------------
# 5) Function that returns multiple values
# ----------------------------

def stats(numbers):
    """
    Returns count, minimum, maximum, and average of a list.
    """
    count = len(numbers)
    minimum = min(numbers)
    maximum = max(numbers)
    average = sum(numbers) / count
    return count, minimum, maximum, average

print("\nStats:", stats(nums))
