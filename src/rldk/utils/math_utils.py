"""Mathematical utilities for safe operations in RLDK.

This module provides safe mathematical operations that prevent division by zero
and other common mathematical errors. All functions return float values for
consistency, even when inputs are integers.

Functions:
    safe_divide: Safely divide two numbers with fallback for invalid denominators
    safe_rate_calculation: Calculate rates (count per unit time) safely
    safe_percentage: Calculate percentages safely
    safe_ratio: Calculate ratios safely

All functions handle:
- Division by zero (returns fallback value)
- Negative denominators (returns fallback value)
- Type conversion (int/float inputs -> float output)
"""

from typing import Union


def safe_divide(numerator: Union[int, float], denominator: Union[int, float], fallback: float = 0.0) -> float:
    """Safely divide two numbers, avoiding division by zero and negative denominators.

    Args:
        numerator: The number to divide
        denominator: The number to divide by
        fallback: Value to return if denominator is zero or negative

    Returns:
        The result of division or fallback value
    """
    if denominator <= 0:
        return fallback
    return numerator / denominator


def safe_rate_calculation(count: Union[int, float], time_interval: Union[int, float], fallback: float = 0.0) -> float:
    """Safely calculate rate (count per unit time), avoiding division by zero and negative denominators.

    Args:
        count: The count or amount
        time_interval: The time interval
        fallback: Value to return if time_interval is zero or negative

    Returns:
        The rate or fallback value
    """
    return safe_divide(count, time_interval, fallback)


def safe_percentage(numerator: Union[int, float], denominator: Union[int, float], fallback: float = 0.0) -> float:
    """Safely calculate percentage, avoiding division by zero and negative denominators.

    Args:
        numerator: The numerator
        denominator: The denominator
        fallback: Value to return if denominator is zero or negative

    Returns:
        The percentage or fallback value
    """
    return safe_divide(numerator * 100, denominator, fallback)


def safe_ratio(numerator: Union[int, float], denominator: Union[int, float], fallback: float = 0.0) -> float:
    """Safely calculate ratio, avoiding division by zero and negative denominators.

    Args:
        numerator: The numerator
        denominator: The denominator
        fallback: Value to return if denominator is zero or negative

    Returns:
        The ratio or fallback value
    """
    return safe_divide(numerator, denominator, fallback)


def try_divide(numerator: Union[int, float], denominator: Union[int, float], fallback: float = 0.0) -> float:
    """Safely divide two numbers, avoiding division by zero and negative denominators.

    This function is consistent with safe_percentage and safe_rate_calculation
    by skipping negative denominators and returning the fallback value.

    Args:
        numerator: The number to divide
        denominator: The number to divide by
        fallback: Value to return if denominator is zero or negative

    Returns:
        The result of division or fallback value
    """
    if denominator <= 0:
        return fallback
    return numerator / denominator


def safe_divide_with_negative_support(numerator: Union[int, float], denominator: Union[int, float], fallback: float = 0.0) -> float:
    """Safely divide two numbers, allowing negative denominators but avoiding division by zero.

    This is an alternative implementation that only checks for zero denominators,
    allowing negative denominators to proceed with normal division.

    Args:
        numerator: The number to divide
        denominator: The number to divide by
        fallback: Value to return if denominator is zero

    Returns:
        The result of division or fallback value
    """
    if denominator == 0:
        return fallback
    return numerator / denominator
