"""Error handling utilities for RLDK."""

import logging
import sys
import traceback
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path
import time
from contextlib import contextmanager
from functools import wraps


class RLDKError(Exception):
    """Base exception for RLDK-specific errors."""
    
    def __init__(self, message: str, suggestion: Optional[str] = None, 
                 error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.suggestion = suggestion
        self.error_code = error_code
        self.details = details or {}


class ValidationError(RLDKError):
    """Raised when input validation fails."""
    pass


class AdapterError(RLDKError):
    """Raised when data adapter operations fail."""
    pass


class EvaluationError(RLDKError):
    """Raised when evaluation operations fail."""
    pass


class TimeoutError(RLDKError):
    """Raised when operations timeout."""
    pass


def format_error_message(error: Exception, context: Optional[str] = None) -> str:
    """Format a user-friendly error message with suggestions."""
    if isinstance(error, RLDKError):
        message = f"❌ {error.message}"
        if error.suggestion:
            message += f"\n\n💡 Suggestion: {error.suggestion}"
        if error.error_code:
            message += f"\n\n🔍 Error Code: {error.error_code}"
        if error.details:
            message += f"\n\n📋 Details: {error.details}"
    else:
        message = f"❌ {str(error)}"
        if context:
            message = f"❌ {context}: {str(error)}"
    
    return message


def log_error_with_context(error: Exception, context: str, logger: Optional[logging.Logger] = None):
    """Log error with full context for debugging."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.error(f"Error in {context}: {error}")
    logger.debug(f"Full traceback:\n{traceback.format_exc()}")


def validate_file_path(path: Union[str, Path], must_exist: bool = True, 
                      file_extensions: Optional[List[str]] = None) -> Path:
    """Validate file path with helpful error messages."""
    try:
        path_obj = Path(path)
    except Exception as e:
        raise ValidationError(
            f"Invalid path format: {path}",
            suggestion="Please provide a valid file or directory path",
            error_code="INVALID_PATH"
        ) from e
    
    if must_exist and not path_obj.exists():
        raise ValidationError(
            f"Path does not exist: {path_obj}",
            suggestion="Please check the path and ensure the file/directory exists",
            error_code="PATH_NOT_FOUND",
            details={"path": str(path_obj), "absolute_path": str(path_obj.absolute())}
        )
    
    if file_extensions and path_obj.is_file():
        if path_obj.suffix not in file_extensions:
            raise ValidationError(
                f"File has unsupported extension: {path_obj.suffix}",
                suggestion=f"Expected one of: {', '.join(file_extensions)}",
                error_code="UNSUPPORTED_EXTENSION",
                details={"file": str(path_obj), "expected_extensions": file_extensions}
            )
    
    return path_obj


def validate_data_format(data: Any, expected_type: type, field_name: str) -> None:
    """Validate data type with helpful error messages."""
    if not isinstance(data, expected_type):
        raise ValidationError(
            f"Invalid {field_name}: expected {expected_type.__name__}, got {type(data).__name__}",
            suggestion=f"Please provide a valid {expected_type.__name__} for {field_name}",
            error_code="INVALID_DATA_TYPE",
            details={"field": field_name, "expected_type": expected_type.__name__, "actual_type": type(data).__name__}
        )


def validate_required_fields(data: Dict[str, Any], required_fields: List[str], context: str = "data") -> None:
    """Validate that required fields are present in data."""
    missing_fields = [field for field in required_fields if field not in data or data[field] is None]
    if missing_fields:
        raise ValidationError(
            f"Missing required fields in {context}: {', '.join(missing_fields)}",
            suggestion=f"Please ensure all required fields are present: {', '.join(required_fields)}",
            error_code="MISSING_REQUIRED_FIELDS",
            details={"missing_fields": missing_fields, "required_fields": required_fields}
        )


@contextmanager
def progress_indicator(operation: str, total: Optional[int] = None):
    """Context manager for showing progress indicators."""
    start_time = time.time()
    print(f"🔄 Starting {operation}...")
    
    try:
        yield
        elapsed = time.time() - start_time
        print(f"✅ {operation} completed in {elapsed:.2f}s")
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ {operation} failed after {elapsed:.2f}s: {e}")
        raise


def with_retry(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator to retry operations with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        print(f"⚠️  Attempt {attempt + 1} failed: {e}")
                        print(f"🔄 Retrying in {current_delay:.1f}s...")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        print(f"❌ All {max_retries + 1} attempts failed")
            
            raise last_exception
        return wrapper
    return decorator


def with_timeout(timeout_seconds: float):
    """Decorator to add timeout to operations."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(
                    f"Operation timed out after {timeout_seconds} seconds",
                    suggestion="Try increasing the timeout or optimizing the operation",
                    error_code="OPERATION_TIMEOUT",
                    details={"timeout_seconds": timeout_seconds}
                )
            
            # Set up timeout
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds))
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Restore old handler
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        
        return wrapper
    return decorator


def handle_graceful_degradation(operation_name: str, fallback_value: Any = None):
    """Decorator for graceful degradation when optional features fail."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"⚠️  {operation_name} failed: {e}")
                print(f"🔄 Continuing with degraded functionality...")
                return fallback_value
        return wrapper
    return decorator


def validate_adapter_source(source: Union[str, Path], expected_formats: List[str]) -> None:
    """Validate that source can be handled by available adapters."""
    source_str = str(source)
    
    # Handle remote URIs (like WandB) separately
    if source_str.startswith("wandb://"):
        # Validate WandB URI format
        try:
            from .validation import validate_wandb_uri
            validate_wandb_uri(source_str)
            return  # WandB URI is valid
        except ValidationError:
            raise  # Re-raise WandB validation errors
        except Exception as e:
            raise ValidationError(
                f"Invalid WandB URI: {source_str}",
                suggestion="Use format: wandb://entity/project/run_id",
                error_code="INVALID_WANDB_URI",
                details={"uri": source_str, "error": str(e)}
            ) from e
    
    # Handle local filesystem paths
    source_path = Path(source)
    
    if not source_path.exists():
        raise ValidationError(
            f"Source does not exist: {source_path}",
            suggestion="Please check the path and ensure the file/directory exists",
            error_code="SOURCE_NOT_FOUND"
        )
    
    # Check if source matches any expected format
    format_hints = []
    if source_path.is_file():
        if source_path.suffix == '.jsonl':
            format_hints.append("JSONL format")
        elif source_path.suffix == '.log':
            format_hints.append("Log format")
    elif source_path.is_dir():
        format_hints.append("Directory with log files")
    
    if not format_hints:
        raise ValidationError(
            f"Unsupported source format: {source_path}",
            suggestion=f"Expected one of: {', '.join(expected_formats)}",
            error_code="UNSUPPORTED_SOURCE_FORMAT",
            details={"source": str(source_path), "expected_formats": expected_formats}
        )


def print_usage_examples(command: str, examples: List[str]) -> None:
    """Print usage examples for a command."""
    print(f"\n📚 Usage examples for '{command}':")
    for i, example in enumerate(examples, 1):
        print(f"  {i}. {example}")


def print_troubleshooting_tips(tips: List[str]) -> None:
    """Print troubleshooting tips."""
    print(f"\n🔧 Troubleshooting tips:")
    for i, tip in enumerate(tips, 1):
        print(f"  {i}. {tip}")


def check_dependencies(required_packages: List[str]) -> None:
    """Check if required packages are installed."""
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        raise ValidationError(
            f"Missing required packages: {', '.join(missing_packages)}",
            suggestion=f"Install missing packages with: pip install {' '.join(missing_packages)}",
            error_code="MISSING_DEPENDENCIES",
            details={"missing_packages": missing_packages}
        )


def safe_operation(operation_name: str, fallback_value: Any = None, 
                  log_errors: bool = True) -> Callable:
    """Decorator for safe operations that can fail gracefully."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger = logging.getLogger(__name__)
                    logger.warning(f"{operation_name} failed: {e}")
                return fallback_value
        return wrapper
    return decorator


def safe_divide(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Safely divide two numbers, avoiding division by zero.
    
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


def safe_rate_calculation(count: float, time_interval: float, fallback: float = 0.0) -> float:
    """Safely calculate rate (count per unit time), avoiding division by zero.
    
    Args:
        count: The count or amount
        time_interval: The time interval
        fallback: Value to return if time_interval is zero
        
    Returns:
        The rate or fallback value
    """
    return safe_divide(count, time_interval, fallback)


def safe_percentage(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Safely calculate percentage, avoiding division by zero.
    
    Args:
        numerator: The numerator
        denominator: The denominator
        fallback: Value to return if denominator is zero
        
    Returns:
        The percentage or fallback value
    """
    return safe_divide(numerator * 100, denominator, fallback)


def safe_ratio(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Safely calculate ratio, avoiding division by zero.
    
    Args:
        numerator: The numerator
        denominator: The denominator
        fallback: Value to return if denominator is zero
        
    Returns:
        The ratio or fallback value
    """
    return safe_divide(numerator, denominator, fallback)