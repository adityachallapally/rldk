"""Mathematical utility functions for RLDK."""

from typing import Union
import math


def try_divide(numerator: Union[int, float], denominator: Union[int, float], 
               fallback: Union[int, float] = 0.0) -> Union[int, float]:
    """Safely divide two numbers, avoiding division by zero and negative denominators.
    
    This function is consistent with safe_percentage and safe_rate_calculation
    in that it explicitly skips negative denominators, returning the fallback value.
    
    Args:
        numerator: The number to divide
        denominator: The number to divide by
        fallback: Value to return if denominator is zero or negative
        
    Returns:
        The result of division or fallback value
        
    Examples:
        >>> try_divide(10, 2)
        5.0
        >>> try_divide(10, 0)
        0.0
        >>> try_divide(10, -2)
        0.0
        >>> try_divide(-10, 2)
        -5.0
    """
    # Skip negative denominators (consistent with safe_percentage and safe_rate_calculation)
    if denominator <= 0:
        return fallback
    return numerator / denominator


def safe_divide_with_negative_support(numerator: Union[int, float], 
                                    denominator: Union[int, float], 
                                    fallback: Union[int, float] = 0.0) -> Union[int, float]:
    """Safely divide two numbers, avoiding division by zero but allowing negative denominators.
    
    This is an alternative implementation that allows negative denominators,
    which differs from try_divide, safe_percentage, and safe_rate_calculation.
    
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


def is_finite_number(value: Union[int, float]) -> bool:
    """Check if a number is finite (not NaN or infinity).
    
    Args:
        value: The number to check
        
    Returns:
        True if the number is finite, False otherwise
    """
    return math.isfinite(value)


def clamp(value: Union[int, float], min_val: Union[int, float], 
          max_val: Union[int, float]) -> Union[int, float]:
    """Clamp a value between min and max bounds.
    
    Args:
        value: The value to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        The clamped value
    """
    return max(min_val, min(value, max_val))