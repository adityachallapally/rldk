"""Standardized error handling utilities for RL Debug Kit."""

import warnings
import logging
from typing import Any, Callable, Optional, Type, Union
from functools import wraps


class RLDebugKitError(Exception):
    """Base exception for RL Debug Kit."""
    pass


class StatisticalError(RLDebugKitError):
    """Error in statistical calculations."""
    pass


class MemoryError(RLDebugKitError):
    """Error related to memory management."""
    pass


class FileOperationError(RLDebugKitError):
    """Error in file operations."""
    pass


class ConfigurationError(RLDebugKitError):
    """Error in configuration."""
    pass


def handle_statistical_error(
    func: Callable,
    default_value: Any = None,
    log_level: str = "warning"
) -> Callable:
    """
    Decorator to handle statistical calculation errors consistently.
    
    Args:
        func: Function to wrap
        default_value: Value to return on error
        log_level: Logging level ('warning', 'error', 'info')
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (ValueError, ArithmeticError, OverflowError) as e:
            error_msg = f"Statistical calculation failed in {func.__name__}: {e}"
            
            if log_level == "warning":
                warnings.warn(error_msg, UserWarning, stacklevel=2)
            elif log_level == "error":
                logging.error(error_msg)
            elif log_level == "info":
                logging.info(error_msg)
            
            return default_value
        except Exception as e:
            error_msg = f"Unexpected error in {func.__name__}: {e}"
            logging.error(error_msg)
            return default_value
    
    return wrapper


def handle_memory_error(
    func: Callable,
    default_value: Any = None,
    cleanup_func: Optional[Callable] = None
) -> Callable:
    """
    Decorator to handle memory-related errors consistently.
    
    Args:
        func: Function to wrap
        default_value: Value to return on error
        cleanup_func: Function to call for cleanup on error
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                error_msg = f"Memory error in {func.__name__}: {e}"
                warnings.warn(error_msg, UserWarning, stacklevel=2)
                
                if cleanup_func:
                    try:
                        cleanup_func()
                    except Exception:
                        pass  # Ignore cleanup errors
                
                return default_value
            else:
                raise  # Re-raise non-memory errors
        except Exception as e:
            error_msg = f"Unexpected error in {func.__name__}: {e}"
            logging.error(error_msg)
            return default_value
    
    return wrapper


def handle_file_error(
    func: Callable,
    default_value: Any = None,
    allowed_errors: tuple = (OSError, UnicodeDecodeError, PermissionError)
) -> Callable:
    """
    Decorator to handle file operation errors consistently.
    
    Args:
        func: Function to wrap
        default_value: Value to return on error
        allowed_errors: Tuple of exception types to catch
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except allowed_errors as e:
            error_msg = f"File operation failed in {func.__name__}: {e}"
            warnings.warn(error_msg, UserWarning, stacklevel=2)
            return default_value
        except Exception as e:
            error_msg = f"Unexpected error in {func.__name__}: {e}"
            logging.error(error_msg)
            return default_value
    
    return wrapper


def safe_execute(
    func: Callable,
    *args,
    default_value: Any = None,
    error_types: tuple = (Exception,),
    log_errors: bool = True,
    **kwargs
) -> Any:
    """
    Safely execute a function with consistent error handling.
    
    Args:
        func: Function to execute
        *args: Positional arguments for function
        default_value: Value to return on error
        error_types: Tuple of exception types to catch
        log_errors: Whether to log errors
        **kwargs: Keyword arguments for function
    
    Returns:
        Function result or default_value on error
    """
    try:
        return func(*args, **kwargs)
    except error_types as e:
        if log_errors:
            error_msg = f"Error in {func.__name__}: {e}"
            warnings.warn(error_msg, UserWarning, stacklevel=2)
        return default_value


def validate_input(
    value: Any,
    expected_type: Type,
    name: str,
    allow_none: bool = False
) -> Any:
    """
    Validate input parameters with consistent error handling.
    
    Args:
        value: Value to validate
        expected_type: Expected type
        name: Parameter name for error messages
        allow_none: Whether None is allowed
    
    Returns:
        Validated value
    
    Raises:
        ValueError: If validation fails
    """
    if value is None and allow_none:
        return value
    
    if not isinstance(value, expected_type):
        raise ValueError(f"{name} must be of type {expected_type.__name__}, got {type(value).__name__}")
    
    return value


def log_performance_warning(
    operation: str,
    threshold: float,
    actual_value: float,
    unit: str = "seconds"
) -> None:
    """
    Log performance warnings consistently.
    
    Args:
        operation: Name of the operation
        threshold: Performance threshold
        actual_value: Actual performance value
        unit: Unit of measurement
    """
    if actual_value > threshold:
        warning_msg = f"Performance warning: {operation} took {actual_value:.2f}{unit}, exceeding threshold of {threshold}{unit}"
        warnings.warn(warning_msg, UserWarning, stacklevel=2)


# Standard error handling decorators for common use cases
def statistical_safe(default_value: Any = None):
    """Decorator for statistical functions with safe error handling."""
    return lambda func: handle_statistical_error(func, default_value)


def memory_safe(default_value: Any = None, cleanup_func: Optional[Callable] = None):
    """Decorator for memory-intensive functions with safe error handling."""
    return lambda func: handle_memory_error(func, default_value, cleanup_func)


def file_safe(default_value: Any = None):
    """Decorator for file operations with safe error handling."""
    return lambda func: handle_file_error(func, default_value)