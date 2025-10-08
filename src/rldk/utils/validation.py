"""Input validation utilities for RLDK."""

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import numpy as np
import pandas as pd

from .error_handling import ValidationError


def validate_file_exists(file_path: Union[str, Path], context: str = "file") -> Path:
    """Validate that a file exists and is accessible."""
    path = Path(file_path)

    if not path.exists():
        raise ValidationError(
            f"{context.capitalize()} does not exist: {path}",
            suggestion="Check that the file path is correct and the file exists",
            error_code="FILE_NOT_FOUND",
            details={"path": str(path), "absolute_path": str(path.absolute())}
        )

    try:
        is_file = path.is_file()
    except TypeError:
        # Some tests patch Path.stat() and return lightweight objects that
        # don't provide the attributes pathlib expects (e.g. ``st_mode``).
        # Fall back to the ``os.path`` implementation so we still perform a
        # meaningful check without tripping over the patched version.
        is_file = os.path.isfile(path)

    if not is_file:
        raise ValidationError(
            f"Path is not a file: {path}",
            suggestion="Ensure the path points to a file, not a directory",
            error_code="NOT_A_FILE",
            details={"path": str(path)}
        )

    # Check read permissions
    if not os.access(path, os.R_OK):
        raise ValidationError(
            f"No read permission for file: {path}",
            suggestion="Check file permissions and ensure you have read access",
            error_code="PERMISSION_DENIED",
            details={"path": str(path)}
        )

    return path


def validate_directory_exists(dir_path: Union[str, Path], context: str = "directory") -> Path:
    """Validate that a directory exists and is accessible."""
    path = Path(dir_path)

    if not path.exists():
        raise ValidationError(
            f"{context.capitalize()} does not exist: {path}",
            suggestion="Check that the directory path is correct and the directory exists",
            error_code="DIRECTORY_NOT_FOUND",
            details={"path": str(path), "absolute_path": str(path.absolute())}
        )

    if not path.is_dir():
        raise ValidationError(
            f"Path is not a directory: {path}",
            suggestion="Ensure the path points to a directory, not a file",
            error_code="NOT_A_DIRECTORY",
            details={"path": str(path)}
        )

    # Check read permissions
    if not os.access(path, os.R_OK):
        raise ValidationError(
            f"No read permission for directory: {path}",
            suggestion="Check directory permissions and ensure you have read access",
            error_code="PERMISSION_DENIED",
            details={"path": str(path)}
        )

    return path


def validate_file_extension(file_path: Union[str, Path],
                          allowed_extensions: List[str],
                          context: str = "file") -> Path:
    """Validate that a file has an allowed extension."""
    path = Path(file_path)

    if path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
        raise ValidationError(
            f"File has unsupported extension: {path.suffix}",
            suggestion=f"Expected one of: {', '.join(allowed_extensions)}",
            error_code="UNSUPPORTED_EXTENSION",
            details={
                "file": str(path),
                "actual_extension": path.suffix,
                "expected_extensions": allowed_extensions
            }
        )

    return path


def validate_json_file(file_path: Union[str, Path], context: str = "JSON file") -> Dict[str, Any]:
    """Validate that a file contains valid JSON."""
    path = validate_file_exists(file_path, context)

    try:
        with open(path) as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValidationError(
                f"JSON file does not contain a dictionary: {path}",
                suggestion="Ensure the JSON file contains a valid JSON object",
                error_code="INVALID_JSON_STRUCTURE",
                details={"path": str(path), "actual_type": type(data).__name__}
            )

        return data

    except json.JSONDecodeError as e:
        raise ValidationError(
            f"Invalid JSON in file: {path}",
            suggestion="Check that the file contains valid JSON syntax",
            error_code="INVALID_JSON",
            details={"path": str(path), "error": str(e)}
        ) from e
    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError(
            f"Failed to read JSON file: {path}",
            suggestion="Check file permissions and encoding",
            error_code="JSON_READ_ERROR",
            details={"path": str(path), "error": str(e)}
        ) from e


def validate_json_file_with_size_check(file_path: Union[str, Path],
                                       context: str = "JSON file",
                                       max_size_mb: float = 100.0) -> Dict[str, Any]:
    """Validate JSON file with size limits (loads entire file into memory)."""
    path = validate_file_exists(file_path, context)

    # Check file size
    file_size = path.stat().st_size
    max_size_bytes = max_size_mb * 1024 * 1024

    if file_size > max_size_bytes:
        raise ValidationError(
            f"JSON file too large: {file_size / 1024 / 1024:.1f} MB > {max_size_mb} MB",
            suggestion=f"File exceeds maximum size limit of {max_size_mb} MB",
            error_code="FILE_TOO_LARGE",
            details={"path": str(path), "file_size_mb": file_size / 1024 / 1024, "max_size_mb": max_size_mb}
        )

    try:
        with open(path) as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValidationError(
                f"JSON file does not contain a dictionary: {path}",
                suggestion="Ensure the JSON file contains a valid JSON object",
                error_code="INVALID_JSON_STRUCTURE",
                details={"path": str(path), "actual_type": type(data).__name__}
            )

        return data

    except json.JSONDecodeError as e:
        raise ValidationError(
            f"Invalid JSON in file: {path}",
            suggestion="Check that the file contains valid JSON syntax",
            error_code="INVALID_JSON",
            details={"path": str(path), "error": str(e)}
        ) from e
    except Exception as e:
        raise ValidationError(
            f"Failed to read JSON file: {path}",
            suggestion="Check file permissions and encoding",
            error_code="JSON_READ_ERROR",
            details={"path": str(path), "error": str(e)}
        ) from e


def validate_json_file_streaming(
    file_path: Union[str, Path],
    context: str = "JSON file",
    max_size_mb: float = 100.0,
) -> Dict[str, Any]:
    """Validate a JSON file with size checks while avoiding circular imports.

    This is a thin wrapper around :func:`validate_json_file_with_size_check` that
    exists for backwards compatibility with the streaming-focused validation
    helpers used in the test suite. It enforces the same size limits and returns
    the parsed JSON content.
    """

    return validate_json_file_with_size_check(
        file_path,
        context=context,
        max_size_mb=max_size_mb,
    )


def validate_jsonl_file(file_path: Union[str, Path], context: str = "JSONL file") -> List[Dict[str, Any]]:
    """Validate that a file contains valid JSONL."""
    path = validate_file_exists(file_path, context)

    try:
        data = []
        with open(path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValidationError(
                            f"JSONL line {line_num} is not a dictionary: {path}",
                            suggestion="Ensure each line contains a valid JSON object",
                            error_code="INVALID_JSONL_STRUCTURE",
                            details={"path": str(path), "line": line_num, "actual_type": type(record).__name__}
                        )
                    data.append(record)
                except json.JSONDecodeError as e:
                    raise ValidationError(
                        f"Invalid JSON on line {line_num} in file: {path}",
                        suggestion="Check that each line contains valid JSON",
                        error_code="INVALID_JSONL_LINE",
                        details={"path": str(path), "line": line_num, "error": str(e)}
                    ) from e

        if not data:
            raise ValidationError(
                f"No valid JSON records found in file: {path}",
                suggestion="Ensure the file contains at least one valid JSON object",
                error_code="EMPTY_JSONL_FILE",
                details={"path": str(path)}
            )

        return data

    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(
            f"Failed to read JSONL file: {path}",
            suggestion="Check file permissions and encoding",
            error_code="JSONL_READ_ERROR",
            details={"path": str(path), "error": str(e)}
        ) from e


def validate_jsonl_file_streaming(file_path: Union[str, Path],
                                 context: str = "JSONL file",
                                 max_size_mb: float = 100.0,
                                 max_lines: int = 1000000) -> Iterator[Dict[str, Any]]:
    """Validate JSONL file with streaming support and size/line limits."""
    path = validate_file_exists(file_path, context)

    # Check file size
    file_size = path.stat().st_size
    max_size_bytes = max_size_mb * 1024 * 1024

    if file_size > max_size_bytes:
        raise ValidationError(
            f"JSONL file too large: {file_size / 1024 / 1024:.1f} MB > {max_size_mb} MB",
            suggestion=f"File exceeds maximum size limit of {max_size_mb} MB",
            error_code="FILE_TOO_LARGE",
            details={"path": str(path), "file_size_mb": file_size / 1024 / 1024, "max_size_mb": max_size_mb}
        )

    try:
        line_count = 0
        with open(path, encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    # Log empty lines for debugging
                    continue

                line_count += 1

                # Check for extremely long lines that could cause memory issues
                if len(line) > 1024 * 1024:  # 1MB line limit
                    raise ValidationError(
                        f"JSONL line {line_num} too long: {len(line)} characters > 1MB",
                        suggestion="Split large records into smaller chunks",
                        error_code="LINE_TOO_LONG",
                        details={"path": str(path), "line": line_num, "line_length": len(line)}
                    )

                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValidationError(
                            f"JSONL line {line_num} is not a dictionary: {path}",
                            suggestion="Ensure each line contains a valid JSON object",
                            error_code="INVALID_JSONL_STRUCTURE",
                            details={"path": str(path), "line": line_num, "actual_type": type(record).__name__}
                        )

                    # Check line limit before yielding to prevent processing over-limit data
                    if line_count > max_lines:
                        raise ValidationError(
                            f"JSONL file has too many lines: {line_count} > {max_lines}",
                            suggestion=f"File exceeds maximum line limit of {max_lines}",
                            error_code="TOO_MANY_LINES",
                            details={"path": str(path), "line_count": line_count, "max_lines": max_lines}
                        )

                    # Yield immediately for true streaming
                    yield record
                except json.JSONDecodeError as e:
                    raise ValidationError(
                        f"Invalid JSON on line {line_num} in file: {path}",
                        suggestion="Check that each line contains valid JSON",
                        error_code="INVALID_JSONL_LINE",
                        details={"path": str(path), "line": line_num, "error": str(e)}
                    ) from e

        if line_count == 0:
            raise ValidationError(
                f"No valid JSON records found in file: {path}",
                suggestion="Ensure the file contains at least one valid JSON object",
                error_code="EMPTY_JSONL_FILE",
                details={"path": str(path)}
            )

    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(
            f"Failed to read JSONL file: {path}",
            suggestion="Check file permissions and encoding",
            error_code="JSONL_READ_ERROR",
            details={"path": str(path), "error": str(e)}
        ) from e


def validate_dataframe(df: pd.DataFrame,
                      required_columns: Optional[List[str]] = None,
                      min_rows: int = 1,
                      context: str = "DataFrame") -> pd.DataFrame:
    """Validate a pandas DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(
            f"{context} must be a pandas DataFrame",
            suggestion="Ensure you're passing a valid DataFrame",
            error_code="INVALID_DATAFRAME_TYPE",
            details={"actual_type": type(df).__name__}
        )

    if len(df) < min_rows:
        raise ValidationError(
            f"{context} has too few rows: {len(df)} < {min_rows}",
            suggestion=f"Ensure the DataFrame has at least {min_rows} rows",
            error_code="INSUFFICIENT_ROWS",
            details={"actual_rows": len(df), "min_rows": min_rows}
        )

    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValidationError(
                f"{context} missing required columns: {missing_columns}",
                suggestion=f"Ensure the DataFrame contains all required columns: {required_columns}",
                error_code="MISSING_COLUMNS",
                details={"missing_columns": missing_columns, "required_columns": required_columns}
            )

    return df


def validate_numeric_range(value: Union[int, float],
                          min_val: Optional[float] = None,
                          max_val: Optional[float] = None,
                          context: str = "value") -> Union[int, float]:
    """Validate that a numeric value is within a range."""
    if not isinstance(value, (int, float)):
        raise ValidationError(
            f"{context} must be numeric, got: {type(value).__name__}",
            suggestion="Provide a valid number",
            error_code="INVALID_NUMERIC_TYPE",
            details={"value": value, "actual_type": type(value).__name__}
        )

    if not np.isfinite(value):
        raise ValidationError(
            f"{context} must be finite, got: {value}",
            suggestion="Provide a finite number",
            error_code="NON_FINITE_VALUE",
            details={"value": value}
        )

    if min_val is not None and value < min_val:
        raise ValidationError(
            f"{context} must be >= {min_val}, got: {value}",
            suggestion=f"Provide a value >= {min_val}",
            error_code="VALUE_TOO_SMALL",
            details={"value": value, "min_value": min_val}
        )

    if max_val is not None and value > max_val:
        raise ValidationError(
            f"{context} must be <= {max_val}, got: {value}",
            suggestion=f"Provide a value <= {max_val}",
            error_code="VALUE_TOO_LARGE",
            details={"value": value, "max_value": max_val}
        )

    return value


def validate_string_not_empty(value: str, context: str = "string") -> str:
    """Validate that a string is not empty."""
    if not isinstance(value, str):
        raise ValidationError(
            f"{context} must be a string, got: {type(value).__name__}",
            suggestion="Provide a valid string",
            error_code="INVALID_STRING_TYPE",
            details={"value": value, "actual_type": type(value).__name__}
        )

    if not value.strip():
        raise ValidationError(
            f"{context} cannot be empty",
            suggestion="Provide a non-empty string",
            error_code="EMPTY_STRING",
            details={"value": value}
        )

    return value.strip()


def validate_choice(value: Any, choices: List[Any], context: str = "value") -> Any:
    """Validate that a value is one of the allowed choices."""
    if value not in choices:
        raise ValidationError(
            f"{context} must be one of {choices}, got: {value}",
            suggestion=f"Choose from: {', '.join(map(str, choices))}",
            error_code="INVALID_CHOICE",
            details={"value": value, "choices": choices}
        )

    return value


def validate_wandb_uri(uri: str) -> Dict[str, str]:
    """Validate a Weights & Biases URI format."""
    if not isinstance(uri, str):
        raise ValidationError(
            "WandB URI must be a string",
            suggestion="Provide a valid WandB URI string",
            error_code="INVALID_URI_TYPE",
            details={"actual_type": type(uri).__name__}
        )

    if not uri.startswith("wandb://"):
        raise ValidationError(
            f"WandB URI must start with 'wandb://', got: {uri}",
            suggestion="Use format: wandb://entity/project/run_id",
            error_code="INVALID_URI_PREFIX",
            details={"uri": uri}
        )

    # Remove wandb:// prefix
    path = uri[8:]
    parts = path.split("/")

    if len(parts) != 3:
        raise ValidationError(
            f"WandB URI must have 3 parts (entity/project/run_id), got {len(parts)}: {uri}",
            suggestion="Use format: wandb://entity/project/run_id",
            error_code="INVALID_URI_FORMAT",
            details={"uri": uri, "parts": parts, "part_count": len(parts)}
        )

    entity, project, run_id = parts

    if not entity.strip():
        raise ValidationError(
            "WandB URI entity cannot be empty",
            suggestion="Provide a valid entity name",
            error_code="EMPTY_ENTITY",
            details={"uri": uri, "entity": entity}
        )

    if not project.strip():
        raise ValidationError(
            "WandB URI project cannot be empty",
            suggestion="Provide a valid project name",
            error_code="EMPTY_PROJECT",
            details={"uri": uri, "project": project}
        )

    if not run_id.strip():
        raise ValidationError(
            "WandB URI run_id cannot be empty",
            suggestion="Provide a valid run ID",
            error_code="EMPTY_RUN_ID",
            details={"uri": uri, "run_id": run_id}
        )

    return {
        "entity": entity,
        "project": project,
        "run_id": run_id
    }


def validate_adapter_type(adapter: str) -> str:
    """Validate adapter type."""
    valid_adapters = ["trl", "openrlhf", "wandb", "custom_jsonl"]
    return validate_choice(adapter, valid_adapters, "adapter type")


def validate_evaluation_suite(suite: str) -> str:
    """Validate evaluation suite name."""
    valid_suites = ["quick", "comprehensive", "safety"]
    return validate_choice(suite, valid_suites, "evaluation suite")


def validate_positive_integer(value: Any, context: str = "value") -> int:
    """Validate that a value is a positive integer."""
    if not isinstance(value, (int, float)):
        try:
            value = int(value)
        except (ValueError, TypeError):
            raise ValidationError(
                f"{context} must be a number, got: {type(value).__name__}",
                suggestion="Provide a valid number",
                error_code="INVALID_NUMERIC_TYPE",
                details={"value": value, "actual_type": type(value).__name__}
            )

    if not isinstance(value, int):
        if value.is_integer():
            value = int(value)
        else:
            raise ValidationError(
                f"{context} must be an integer, got: {value}",
                suggestion="Provide a whole number",
                error_code="NON_INTEGER_VALUE",
                details={"value": value}
            )

    if value <= 0:
        raise ValidationError(
            f"{context} must be positive, got: {value}",
            suggestion="Provide a positive integer",
            error_code="NON_POSITIVE_VALUE",
            details={"value": value}
        )

    return value


def validate_non_negative_integer(value: Any, context: str = "value") -> int:
    """Validate that a value is a non-negative integer."""
    if not isinstance(value, (int, float)):
        try:
            value = int(value)
        except (ValueError, TypeError):
            raise ValidationError(
                f"{context} must be a number, got: {type(value).__name__}",
                suggestion="Provide a valid number",
                error_code="INVALID_NUMERIC_TYPE",
                details={"value": value, "actual_type": type(value).__name__}
            )

    if not isinstance(value, int):
        if value.is_integer():
            value = int(value)
        else:
            raise ValidationError(
                f"{context} must be an integer, got: {value}",
                suggestion="Provide a whole number",
                error_code="NON_INTEGER_VALUE",
                details={"value": value}
            )

    if value < 0:
        raise ValidationError(
            f"{context} must be non-negative, got: {value}",
            suggestion="Provide a non-negative integer",
            error_code="NEGATIVE_VALUE",
            details={"value": value}
        )

    return value


def validate_probability(value: Any, context: str = "probability") -> float:
    """Validate that a value is a valid probability (0.0 to 1.0)."""
    return validate_numeric_range(value, 0.0, 1.0, context)


def validate_optional_string(value: Any, context: str = "string") -> Optional[str]:
    """Validate an optional string (None or non-empty string)."""
    if value is None:
        return None

    return validate_string_not_empty(value, context)


def validate_optional_positive_integer(value: Any, context: str = "value") -> Optional[int]:
    """Validate an optional positive integer (None or positive integer)."""
    if value is None:
        return None

    return validate_positive_integer(value, context)


def validate_file_size(file_path: Union[str, Path], max_size_mb: float = 100.0) -> Path:
    """Validate that a file is not too large."""
    path = validate_file_exists(file_path)

    file_size = path.stat().st_size
    max_size_bytes = max_size_mb * 1024 * 1024

    if file_size > max_size_bytes:
        raise ValidationError(
            f"File too large: {file_size / (1024*1024):.1f}MB > {max_size_mb}MB",
            suggestion="Use a smaller file or increase the size limit",
            error_code="FILE_TOO_LARGE",
            details={
                "file": str(path),
                "file_size_mb": file_size / (1024*1024),
                "max_size_mb": max_size_mb
            }
        )

    return path


def validate_data_quality(df: pd.DataFrame,
                         required_columns: List[str],
                         max_missing_ratio: float = 0.5) -> pd.DataFrame:
    """Validate data quality in a DataFrame."""
    df = validate_dataframe(df, required_columns)

    for col in required_columns:
        if col in df.columns:
            missing_ratio = df[col].isna().sum() / len(df)
            if missing_ratio > max_missing_ratio:
                raise ValidationError(
                    f"Column '{col}' has too many missing values: {missing_ratio:.1%} > {max_missing_ratio:.1%}",
                    suggestion="Check data quality and missing value handling",
                    error_code="POOR_DATA_QUALITY",
                    details={
                        "column": col,
                        "missing_ratio": missing_ratio,
                        "max_missing_ratio": max_missing_ratio,
                        "missing_count": df[col].isna().sum(),
                        "total_count": len(df)
                    }
                )

    return df


def validate_with_custom_validator(value: Any,
                                 validator: Callable[[Any], bool],
                                 error_message: str,
                                 context: str = "value") -> Any:
    """Validate a value using a custom validator function."""
    try:
        if not validator(value):
            raise ValidationError(
                f"{context} validation failed: {error_message}",
                suggestion="Check the value and try again",
                error_code="CUSTOM_VALIDATION_FAILED",
                details={"value": value, "context": context}
            )
    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError(
            f"Custom validation error for {context}: {e}",
            suggestion="Check the validator function and value",
            error_code="VALIDATOR_ERROR",
            details={"value": value, "context": context, "error": str(e)}
        ) from e

    return value
