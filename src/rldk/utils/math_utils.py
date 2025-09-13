"""Mathematical utilities for robust division and rate calculations."""

import logging
from typing import Tuple, Union, Literal, List

logger = logging.getLogger(__name__)


def try_divide(
    numerator: Union[int, float], 
    denominator: Union[int, float], 
    *, 
    on_zero: Literal["skip", "zero", "nan"] = "skip",
    fallback: float = float("nan")
) -> Tuple[float, bool]:
    """
    Safely divide two numbers with configurable zero handling.
    
    Args:
        numerator: The numerator value
        denominator: The denominator value
        on_zero: How to handle zero denominators:
            - "skip": Return fallback and mark as unused
            - "zero": Return 0.0 and mark as used
            - "nan": Return NaN and mark as used
        fallback: Value to return when skipping (default: NaN)
        
    Returns:
        Tuple of (result, used) where:
        - result: The division result or fallback
        - used: Whether the sample was used in calculations
        
    Examples:
        >>> try_divide(10, 2)
        (5.0, True)
        >>> try_divide(10, 0, on_zero="skip")
        (nan, False)
        >>> try_divide(10, 0, on_zero="zero")
        (0.0, True)
        >>> try_divide(10, 0, on_zero="nan")
        (nan, True)
    """
    if denominator == 0:
        if on_zero == "skip":
            logger.debug(f"Skipping division by zero: {numerator} / {denominator}")
            return fallback, False
        elif on_zero == "zero":
            logger.debug(f"Zero division result: {numerator} / {denominator} = 0.0")
            return 0.0, True
        elif on_zero == "nan":
            logger.debug(f"NaN division result: {numerator} / {denominator} = NaN")
            return float("nan"), True
        else:
            raise ValueError(f"Invalid on_zero value: {on_zero}")
    else:
        # Handle both positive and negative denominators normally
        return numerator / denominator, True


def nan_aware_mean(values: List[float]) -> float:
    """
    Calculate mean ignoring NaN values.
    
    Args:
        values: List of numeric values that may contain NaN
        
    Returns:
        Mean of non-NaN values, or NaN if all values are NaN
    """
    if not values:
        return float("nan")
    
    valid_values = [v for v in values if not (isinstance(v, float) and (v != v))]  # v != v checks for NaN
    if not valid_values:
        return float("nan")
    
    return sum(valid_values) / len(valid_values)


def nan_aware_std(values: List[float], ddof: int = 1) -> float:
    """
    Calculate standard deviation ignoring NaN values.
    
    Args:
        values: List of numeric values that may contain NaN
        ddof: Delta degrees of freedom (default: 1 for sample std)
        
    Returns:
        Standard deviation of non-NaN values, or NaN if insufficient data
    """
    if not values:
        return float("nan")
    
    valid_values = [v for v in values if not (isinstance(v, float) and (v != v))]  # v != v checks for NaN
    if len(valid_values) <= ddof:
        return float("nan")
    
    mean_val = sum(valid_values) / len(valid_values)
    variance = sum((v - mean_val) ** 2 for v in valid_values) / (len(valid_values) - ddof)
    return variance ** 0.5


def nan_aware_median(values: List[float]) -> float:
    """
    Calculate median ignoring NaN values.
    
    Args:
        values: List of numeric values that may contain NaN
        
    Returns:
        Median of non-NaN values, or NaN if all values are NaN
    """
    if not values:
        return float("nan")
    
    valid_values = [v for v in values if not (isinstance(v, float) and (v != v))]  # v != v checks for NaN
    if not valid_values:
        return float("nan")
    
    sorted_values = sorted(valid_values)
    n = len(sorted_values)
    if n % 2 == 0:
        return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
    else:
        return sorted_values[n // 2]


def safe_percentage(numerator: Union[int, float], denominator: Union[int, float], **kwargs) -> Tuple[float, bool, str]:
    """
    Calculate percentage with robust zero handling and provenance tracking.
    
    Args:
        numerator: The numerator value
        denominator: The denominator value
        **kwargs: Additional arguments passed to try_divide
        
    Returns:
        Tuple of (percentage, used, reason) where:
        - percentage: The calculated percentage or fallback
        - used: Whether the sample was used in calculations
        - reason: String explaining why sample was or wasn't used
    """
    if denominator == 0:
        if kwargs.get("on_zero") == "skip":
            return float("nan"), False, "zero_denominator_skipped"
        elif kwargs.get("on_zero") == "zero":
            return 0.0, True, "zero_denominator_as_zero"
        elif kwargs.get("on_zero") == "nan":
            return float("nan"), True, "zero_denominator_as_nan"
        else:
            return float("nan"), False, "zero_denominator_skipped"
    elif denominator < 0:
        return float("nan"), False, "negative_denominator_skipped"
    else:
        result, used = try_divide(numerator, denominator, **kwargs)
        return result * 100, used, "calculated_successfully"


def safe_rate(numerator: Union[int, float], denominator: Union[int, float], **kwargs) -> Tuple[float, bool, str]:
    """
    Calculate rate with robust zero handling and provenance tracking.
    
    Args:
        numerator: The numerator value
        denominator: The denominator value
        **kwargs: Additional arguments passed to try_divide
        
    Returns:
        Tuple of (rate, used, reason) where:
        - rate: The calculated rate or fallback
        - used: Whether the sample was used in calculations
        - reason: String explaining why sample was or wasn't used
    """
    if denominator == 0:
        if kwargs.get("on_zero") == "skip":
            return float("nan"), False, "zero_denominator_skipped"
        elif kwargs.get("on_zero") == "zero":
            return 0.0, True, "zero_denominator_as_zero"
        elif kwargs.get("on_zero") == "nan":
            return float("nan"), True, "zero_denominator_as_nan"
        else:
            return float("nan"), False, "zero_denominator_skipped"
    elif denominator < 0:
        return float("nan"), False, "negative_denominator_skipped"
    else:
        result, used = try_divide(numerator, denominator, **kwargs)
        return result, used, "calculated_successfully"