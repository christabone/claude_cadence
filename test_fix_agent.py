#!/usr/bin/env python3
"""
Test file for fix agent - intentionally contains issues for testing.
"""

def calculate_average(numbers):
    """Calculate the average of a list of numbers."""
    # Issue 1: No check for empty list (will cause division by zero)
    total = sum(numbers)
    count = len(numbers)
    return total / count  # This will fail if count is 0


def process_user_input(user_input):
    """Process user input without validation."""
    # Issue 2: No input validation (potential security issue)
    # Issue 3: Using eval() is dangerous
    result = eval(user_input)
    return result


def main():
    # Test the functions
    nums = [1, 2, 3, 4, 5]
    avg = calculate_average(nums)
    print(f"Average: {avg}")

    # This would fail:
    # empty_avg = calculate_average([])

    # Dangerous:
    user_data = "2 + 2"
    result = process_user_input(user_data)
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
