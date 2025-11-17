"""
basics_1.py
First Python practice file for AI/ML foundations.
"""

# 1) Simple greeting
print("Hello, future AI/ML Engineer!")

# 2) Basic variables
name = "Vishnu Priya"
age = 25
is_student = True

print("Name:", name)
print("Age:", age)
print("Is student:", is_student)

# 3) List of numbers – sum and average
numbers = [10, 20, 35, 50, 85]

total = sum(numbers)
average = total / len(numbers)

print("\nNumbers:", numbers)
print("Total:", total)
print("Average:", average)

# 4) Dictionary representing a student profile
student = {
    "name": "Vishnu Priya",
    "program": "MS in Technology Management",
    "goal": "AI/ML Engineer",
    "country": "USA"
}

print("\nStudent Profile:")
for key, value in student.items():
    print(f"  {key}: {value}")

# 5) Function to summarize a list of numbers
def summarize_numbers(nums):
    """
    Returns the count, minimum, maximum, and average of a list of numbers.
    """
    count = len(nums)
    minimum = min(nums)
    maximum = max(nums)
    avg = sum(nums) / count
    return count, minimum, maximum, avg


stats = summarize_numbers(numbers)
print("\nSummary stats for numbers:")
print("  Count:", stats[0])
print("  Min:", stats[1])
print("  Max:", stats[2])
print("  Avg:", stats[3])
