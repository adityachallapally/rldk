# Error Handling in RLDK

This document describes the comprehensive error handling system implemented in RLDK to address the issues identified in the error handling analysis.

## Overview

The RLDK error handling system provides:
- **Clear error messages** with suggestions and context
- **Input validation** with helpful feedback
- **Graceful degradation** when optional features fail
- **Progress indication** for long-running operations
- **Comprehensive logging** for debugging

## Error Types

### RLDKError
Base exception class for all RLDK-specific errors.

```python
class RLDKError(Exception):
    def __init__(self, message: str, suggestion: Optional[str] = None, 
                 error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None)
```

### ValidationError
Raised when input validation fails.

```python
# Example usage
try:
    validate_file_path("/nonexistent/file.jsonl")
except ValidationError as e:
    print(format_error_message(e))
```

### AdapterError
Raised when data adapter operations fail.

```python
# Example usage
try:
    adapter = TRLAdapter("/path/to/data")
    df = adapter.load()
except AdapterError as e:
    print(format_error_message(e))
```

### EvaluationError
Raised when evaluation operations fail.

```python
# Example usage
try:
    result = run_evaluation_suite(data, "quick")
except EvaluationError as e:
    print(format_error_message(e))
```

### RLDKTimeoutError
Raised when operations timeout.

```python
# Example usage
@with_timeout(300)  # 5 minutes
def long_operation():
    # ... operation code
    pass
```

## Error Message Formatting

All error messages follow a consistent format:

```
❌ Error message

💡 Suggestion: How to fix the issue

🔍 Error Code: ERROR_CODE

📋 Details: Additional context information
```

### Example Error Message

```
❌ File has unsupported extension: .txt

💡 Suggestion: Expected one of: .jsonl, .log

🔍 Error Code: UNSUPPORTED_EXTENSION

📋 Details: {'file': '/path/to/file.txt', 'expected_extensions': ['.jsonl', '.log']}
```

## Input Validation

### File Validation

```python
from rldk.utils.validation import validate_file_exists, validate_file_extension

# Validate file exists and has correct extension
file_path = validate_file_exists("/path/to/data.jsonl")
file_path = validate_file_extension(file_path, [".jsonl", ".log"])
```

### Data Validation

```python
from rldk.utils.validation import validate_dataframe, validate_positive_integer

# Validate DataFrame
df = validate_dataframe(data, required_columns=["step", "reward_mean"])

# Validate numeric input
value = validate_positive_integer(42, "sample_size")
```

### WandB URI Validation

```python
from rldk.utils.validation import validate_wandb_uri

# Validate WandB URI format
uri_parts = validate_wandb_uri("wandb://entity/project/run_id")
```

## Progress Indication

### Basic Progress Bar

```python
from rldk.utils.progress import progress_bar

with progress_bar(100, "Processing items") as bar:
    for i in range(100):
        # ... process item
        bar.update(1)
```

### Spinner for Indeterminate Operations

```python
from rldk.utils.progress import spinner

with spinner("Loading data"):
    # ... long-running operation
    pass
```

### Task Tracking

```python
from rldk.utils.progress import track_tasks

with track_tasks(5, "Processing files") as tracker:
    for i, file in enumerate(files):
        with tracker.start_task(f"file_{i}"):
            # ... process file
            pass
```

## Graceful Degradation

### Safe Operations

```python
from rldk.utils.error_handling import safe_operation

@safe_operation("Optional feature", fallback_value=None)
def optional_feature():
    # ... feature that might fail
    pass
```

### Retry with Exponential Backoff

```python
from rldk.utils.error_handling import with_retry

@with_retry(max_retries=3, delay=1.0, backoff=2.0)
def unreliable_operation():
    # ... operation that might fail
    pass
```

### Timeout Handling

```python
from rldk.utils.error_handling import with_timeout

@with_timeout(300)  # 5 minutes
def long_operation():
    # ... operation that might timeout
    pass
```

## CLI Error Handling

### Command-Level Error Handling

All CLI commands now include comprehensive error handling:

```python
@app.command()
def my_command(input_file: Path):
    try:
        # Validate input
        validate_file_exists(input_file)
        
        # Process with progress indication
        with timed_operation("Processing"):
            result = process_data(input_file)
        
        # Display results
        print_operation_status("Processing", "success")
        
    except ValidationError as e:
        typer.echo(format_error_message(e), err=True)
        print_usage_examples("my-command", ["example usage"])
        print_troubleshooting_tips(["tip1", "tip2"])
        raise typer.Exit(1)
    except Exception as e:
        log_error_with_context(e, "my-command")
        typer.echo(format_error_message(e, "Command failed"), err=True)
        raise typer.Exit(1)
```

### Error Recovery

Commands now provide:
- **Usage examples** when validation fails
- **Troubleshooting tips** for common issues
- **Graceful degradation** when optional features fail
- **Progress indication** for long operations

## Adapter Error Handling

### Base Adapter Improvements

```python
class BaseAdapter(ABC):
    def _safe_load(self) -> pd.DataFrame:
        """Safely load data with error handling."""
        try:
            return self.load()
        except Exception as e:
            raise AdapterError(
                f"Failed to load data from {self.source}: {e}",
                suggestion="Check that the source contains valid data",
                error_code="LOAD_FAILED"
            ) from e
```

### File Error Handling

```python
def _handle_file_error(self, file_path: Path, operation: str) -> None:
    """Handle file operation errors with helpful messages."""
    if not file_path.exists():
        raise AdapterError(
            f"File not found: {file_path}",
            suggestion="Check that the file path is correct",
            error_code="FILE_NOT_FOUND"
        )
    # ... additional error handling
```

## Evaluation Error Handling

### Graceful Degradation

```python
def run_evaluation_suite(data, suite_name):
    failed_evaluations = []
    
    for eval_name, eval_func in evaluations.items():
        try:
            result = eval_func(data)
            # ... process result
        except Exception as e:
            logger.error(f"Evaluation {eval_name} failed: {e}")
            failed_evaluations.append(eval_name)
            # Continue with other evaluations
    
    # Check if too many evaluations failed
    if len(failed_evaluations) > len(evaluations) * 0.5:
        raise EvaluationError(
            f"Too many evaluations failed: {len(failed_evaluations)}/{len(evaluations)}",
            suggestion="Check your data format and evaluation requirements"
        )
```

## Best Practices

### 1. Always Provide Context

```python
# Good
raise ValidationError(
    f"File not found: {file_path}",
    suggestion="Check that the file path is correct",
    error_code="FILE_NOT_FOUND",
    details={"path": str(file_path), "absolute_path": str(file_path.absolute())}
)

# Bad
raise ValueError("File not found")
```

### 2. Use Appropriate Error Types

```python
# Use ValidationError for input validation
if not isinstance(data, pd.DataFrame):
    raise ValidationError("Data must be a DataFrame")

# Use AdapterError for data loading issues
if not adapter.can_handle():
    raise AdapterError("Adapter cannot handle source")

# Use EvaluationError for evaluation failures
if too_many_failures:
    raise EvaluationError("Too many evaluations failed")
```

### 3. Provide Helpful Suggestions

```python
# Good
raise ValidationError(
    "Invalid adapter type: custom",
    suggestion="Use one of: trl, openrlhf, wandb, custom_jsonl",
    error_code="INVALID_ADAPTER_TYPE"
)

# Bad
raise ValueError("Invalid adapter")
```

### 4. Log Errors with Context

```python
try:
    result = risky_operation()
except Exception as e:
    log_error_with_context(e, "risky_operation")
    raise
```

### 5. Use Progress Indication

```python
# For long operations
with timed_operation("Data processing"):
    result = process_large_dataset()

# For iterations
with progress_bar(len(items), "Processing items") as bar:
    for item in items:
        process_item(item)
        bar.update(1)
```

## Error Codes

Common error codes used throughout RLDK:

- `FILE_NOT_FOUND`: File does not exist
- `PERMISSION_DENIED`: Insufficient permissions
- `INVALID_FORMAT`: Invalid data format
- `VALIDATION_FAILED`: Input validation failed
- `ADAPTER_ERROR`: Data adapter error
- `EVALUATION_FAILED`: Evaluation error
- `TIMEOUT`: Operation timed out
- `MISSING_DEPENDENCIES`: Required packages not installed

## Troubleshooting

### Common Issues and Solutions

1. **"Cannot handle source" error**
   - Check that the source path exists
   - Verify the data format matches the adapter
   - Try specifying the adapter type explicitly

2. **"No data found" error**
   - Ensure the source contains valid training data
   - Check file permissions and encoding
   - Verify the data format is supported

3. **"Validation failed" error**
   - Check input parameters and file paths
   - Ensure required columns are present
   - Verify data types and ranges

4. **"Timeout" error**
   - Increase timeout value if appropriate
   - Check system resources and performance
   - Consider using smaller datasets

### Getting Help

When encountering errors:

1. Check the error message and suggestion
2. Review the error code and details
3. Try the troubleshooting tips provided
4. Use the `--verbose` flag for detailed output
5. Check the logs for additional context

## Migration Guide

### Updating Existing Code

1. **Replace generic exceptions** with specific RLDK error types
2. **Add input validation** using the validation utilities
3. **Include progress indication** for long operations
4. **Provide helpful error messages** with suggestions
5. **Use graceful degradation** for optional features

### Example Migration

```python
# Before
def load_data(file_path):
    if not os.path.exists(file_path):
        raise ValueError("File not found")
    return pd.read_json(file_path)

# After
def load_data(file_path):
    try:
        path = validate_file_exists(file_path, "data file")
        path = validate_file_extension(path, [".json", ".jsonl"])
        
        with spinner("Loading data"):
            return pd.read_json(path)
    except ValidationError as e:
        raise AdapterError(
            f"Failed to load data: {e}",
            suggestion="Check file path and format",
            error_code="LOAD_FAILED"
        ) from e
```