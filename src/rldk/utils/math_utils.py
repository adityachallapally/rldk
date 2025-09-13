"""Mathematical utilities for safe operations in RLDK."""

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