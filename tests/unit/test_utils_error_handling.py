#!/usr/bin/env python3
"""Unit tests for rldk.utils.error_handling module."""

import json
import logging
import signal
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# Import the module under test
from rldk.utils.error_handling import (
    AdapterError,
    EvaluationError,
    RLDKError,
    RLDKTimeoutError,
    ValidationError,
    check_dependencies,
    format_error_message,
    handle_graceful_degradation,
    log_error_with_context,
    print_troubleshooting_tips,
    print_usage_examples,
    progress_indicator,
    safe_operation,
    sanitize_path,
    validate_adapter_source,
    validate_data_format,
    validate_file_path,
    validate_required_fields,
    with_retry,
)
from rldk.utils.runtime import with_timeout
from rldk.utils.validation import (
    validate_json_file_streaming,
    validate_jsonl_file_streaming,
)


class TestExceptionClasses:
    """Test exception classes."""

    def test_rldk_error_basic(self):
        """Test basic RLDKError functionality."""
        error = RLDKError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.suggestion is None
        assert error.error_code is None
        assert error.details == {}

    def test_rldk_error_with_all_fields(self):
        """Test RLDKError with all fields."""
        details = {"key": "value"}
        error = RLDKError(
            "Test error",
            suggestion="Test suggestion",
            error_code="TEST_ERROR",
            details=details
        )
        assert error.message == "Test error"
        assert error.suggestion == "Test suggestion"
        assert error.error_code == "TEST_ERROR"
        assert error.details == details

    def test_validation_error(self):
        """Test ValidationError inheritance."""
        error = ValidationError("Validation failed")
        assert isinstance(error, RLDKError)
        assert str(error) == "Validation failed"

    def test_adapter_error(self):
        """Test AdapterError inheritance."""
        error = AdapterError("Adapter failed")
        assert isinstance(error, RLDKError)
        assert str(error) == "Adapter failed"

    def test_evaluation_error(self):
        """Test EvaluationError inheritance."""
        error = EvaluationError("Evaluation failed")
        assert isinstance(error, RLDKError)
        assert str(error) == "Evaluation failed"

    def test_timeout_error(self):
        """Test RLDKTimeoutError inheritance."""
        error = RLDKTimeoutError("Operation timed out")
        assert isinstance(error, RLDKError)
        assert str(error) == "Operation timed out"


class TestErrorFormatting:
    """Test error formatting functions."""

    def test_format_error_message_rldk_error(self):
        """Test formatting RLDKError message."""
        error = RLDKError(
            "Test error",
            suggestion="Test suggestion",
            error_code="TEST_ERROR",
            details={"key": "value"}
        )

        message = format_error_message(error)

        assert "❌ Test error" in message
        assert "💡 Suggestion: Test suggestion" in message
        assert "🔍 Error Code: TEST_ERROR" in message
        assert "📋 Details: {'key': 'value'}" in message

    def test_format_error_message_rldk_error_minimal(self):
        """Test formatting RLDKError with minimal fields."""
        error = RLDKError("Test error")

        message = format_error_message(error)

        assert "❌ Test error" in message
        assert "💡 Suggestion:" not in message
        assert "🔍 Error Code:" not in message
        assert "📋 Details:" not in message

    def test_format_error_message_generic_error(self):
        """Test formatting generic error message."""
        error = ValueError("Generic error")

        message = format_error_message(error)

        assert "❌ Generic error" in message

    def test_format_error_message_generic_error_with_context(self):
        """Test formatting generic error message with context."""
        error = ValueError("Generic error")

        message = format_error_message(error, context="Test context")

        assert "❌ Test context: Generic error" in message

    def test_log_error_with_context(self):
        """Test logging error with context."""
        error = ValueError("Test error")

        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            log_error_with_context(error, "Test context")

            mock_logger.error.assert_called_once()
            mock_logger.debug.assert_called_once()


class TestFilePathValidation:
    """Test file path validation functions."""

    def test_validate_file_path_success(self, temp_dir):
        """Test successful file path validation."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        result = validate_file_path(test_file)
        assert result == test_file
        assert isinstance(result, Path)

    def test_validate_file_path_invalid_format(self):
        """Test file path validation with invalid format."""
        with pytest.raises(ValidationError) as exc_info:
            validate_file_path(None)

        error = exc_info.value
        assert "Invalid path format" in str(error)
        assert error.error_code == "INVALID_PATH"

    def test_validate_file_path_not_exists(self, temp_dir):
        """Test file path validation when file doesn't exist."""
        test_file = temp_dir / "nonexistent.txt"

        with pytest.raises(ValidationError) as exc_info:
            validate_file_path(test_file)

        error = exc_info.value
        assert "does not exist" in str(error)
        assert error.error_code == "PATH_NOT_FOUND"

    def test_validate_file_path_not_exists_optional(self, temp_dir):
        """Test file path validation when file doesn't exist but not required."""
        test_file = temp_dir / "nonexistent.txt"

        # Should not raise exception when must_exist=False
        result = validate_file_path(test_file, must_exist=False)
        assert result == test_file

    def test_validate_file_path_extension_success(self, temp_dir):
        """Test file path validation with correct extension."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        result = validate_file_path(test_file, file_extensions=[".txt", ".log"])
        assert result == test_file

    def test_validate_file_path_extension_invalid(self, temp_dir):
        """Test file path validation with invalid extension."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        with pytest.raises(ValidationError) as exc_info:
            validate_file_path(test_file, file_extensions=[".log", ".json"])

        error = exc_info.value
        assert "unsupported extension" in str(error)
        assert error.error_code == "UNSUPPORTED_EXTENSION"


class TestPathSanitization:
    """Test path sanitization functions."""

    def test_sanitize_path_success(self, temp_dir):
        """Test successful path sanitization."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        result = sanitize_path(test_file)
        assert result == test_file.resolve()
        assert isinstance(result, Path)

    def test_sanitize_path_traversal_detected(self):
        """Test path sanitization detects traversal attempts."""
        with pytest.raises(ValidationError) as exc_info:
            sanitize_path("../etc/passwd")

        error = exc_info.value
        assert "path traversal attempt" in str(error).lower()
        assert error.error_code == "PATH_TRAVERSAL_DETECTED"

    def test_sanitize_path_absolute_path_allowed(self):
        """Test path sanitization allows legitimate absolute paths."""
        # Should not raise an error for legitimate absolute paths
        result = sanitize_path("/etc/passwd")
        assert isinstance(result, Path)
        assert str(result) == "/etc/passwd"

    def test_sanitize_path_windows_path_allowed(self):
        """Test path sanitization allows Windows-style paths."""
        # Should not raise an error for Windows paths
        result = sanitize_path("C:\\Users\\test\\file.txt")
        assert isinstance(result, Path)
        # Path will be normalized by Path.resolve()

    def test_sanitize_path_with_base_path_success(self, temp_dir):
        """Test path sanitization with base path restriction."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        result = sanitize_path(test_file, base_path=temp_dir)
        assert result == test_file.resolve()

    def test_sanitize_path_outside_base_path(self, temp_dir):
        """Test path sanitization rejects paths outside base path."""
        outside_file = temp_dir.parent / "outside.txt"

        with pytest.raises(ValidationError) as exc_info:
            sanitize_path(outside_file, base_path=temp_dir)

        error = exc_info.value
        assert "path outside allowed directory" in str(error).lower()
        assert error.error_code == "PATH_OUTSIDE_BASE"

    def test_validate_file_path_with_sanitization(self, temp_dir):
        """Test validate_file_path uses sanitization."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        # Should work with valid path
        result = validate_file_path(test_file, base_path=temp_dir)
        assert result == test_file.resolve()

        # Should reject traversal attempts
        with pytest.raises(ValidationError) as exc_info:
            validate_file_path("../etc/passwd", base_path=temp_dir)

        error = exc_info.value
        assert error.error_code == "PATH_TRAVERSAL_DETECTED"


class TestStreamingValidation:
    """Test streaming validation functions."""

    def test_validate_json_file_streaming_success(self, temp_dir):
        """Test successful JSON file streaming validation."""
        test_file = temp_dir / "test.json"
        test_data = {"key": "value", "number": 42}
        test_file.write_text(json.dumps(test_data))

        result = validate_json_file_streaming(test_file)
        assert result == test_data

    def test_validate_json_file_streaming_too_large(self, temp_dir):
        """Test JSON file streaming validation with size limit."""
        test_file = temp_dir / "large.json"
        # Create a large JSON file (simulate)
        large_data = {"data": "x" * 1024 * 1024}  # 1MB of data
        test_file.write_text(json.dumps(large_data))

        with pytest.raises(ValidationError) as exc_info:
            validate_json_file_streaming(test_file, max_size_mb=0.5)  # 0.5MB limit

        error = exc_info.value
        assert "file too large" in str(error).lower()
        assert error.error_code == "FILE_TOO_LARGE"

    def test_validate_jsonl_file_streaming_success(self, temp_dir):
        """Test successful JSONL file streaming validation."""
        test_file = temp_dir / "test.jsonl"
        test_data = [{"id": 1, "name": "test1"}, {"id": 2, "name": "test2"}]
        test_file.write_text("\n".join(json.dumps(record) for record in test_data))

        result = list(validate_jsonl_file_streaming(test_file))
        assert result == test_data

    def test_validate_jsonl_file_streaming_too_many_lines(self, temp_dir):
        """Test JSONL file streaming validation with line limit."""
        test_file = temp_dir / "large.jsonl"
        # Create a file with many lines
        lines = [json.dumps({"id": i}) for i in range(1000001)]  # 1M+ lines
        test_file.write_text("\n".join(lines))

        with pytest.raises(ValidationError) as exc_info:
            list(validate_jsonl_file_streaming(test_file, max_lines=1000000))

        error = exc_info.value
        assert "too many lines" in str(error).lower()
        assert error.error_code == "TOO_MANY_LINES"

    def test_validate_jsonl_file_streaming_empty_file(self, temp_dir):
        """Test JSONL file streaming validation with empty file."""
        test_file = temp_dir / "empty.jsonl"
        test_file.write_text("")

        with pytest.raises(ValidationError) as exc_info:
            list(validate_jsonl_file_streaming(test_file))

        error = exc_info.value
        assert "no valid json records" in str(error).lower()
        assert error.error_code == "EMPTY_JSONL_FILE"


class TestDataValidation:
    """Test data validation functions."""

    def test_validate_data_format_success(self):
        """Test successful data format validation."""
        # Should not raise exception
        validate_data_format("test", str, "field_name")
        validate_data_format(123, int, "field_name")
        validate_data_format([1, 2, 3], list, "field_name")

    def test_validate_data_format_invalid_type(self):
        """Test data format validation with invalid type."""
        with pytest.raises(ValidationError) as exc_info:
            validate_data_format("test", int, "field_name")

        error = exc_info.value
        assert "Invalid field_name" in str(error)
        assert error.error_code == "INVALID_DATA_TYPE"

    def test_validate_required_fields_success(self):
        """Test successful required fields validation."""
        data = {"field1": "value1", "field2": "value2", "field3": "value3"}
        required_fields = ["field1", "field2"]

        # Should not raise exception
        validate_required_fields(data, required_fields)

    def test_validate_required_fields_missing(self):
        """Test required fields validation with missing fields."""
        data = {"field1": "value1"}
        required_fields = ["field1", "field2", "field3"]

        with pytest.raises(ValidationError) as exc_info:
            validate_required_fields(data, required_fields)

        error = exc_info.value
        assert "Missing required fields" in str(error)
        assert error.error_code == "MISSING_REQUIRED_FIELDS"
        assert "field2" in error.details["missing_fields"]
        assert "field3" in error.details["missing_fields"]

    def test_validate_required_fields_none_values(self):
        """Test required fields validation with None values."""
        data = {"field1": "value1", "field2": None, "field3": "value3"}
        required_fields = ["field1", "field2", "field3"]

        with pytest.raises(ValidationError) as exc_info:
            validate_required_fields(data, required_fields)

        error = exc_info.value
        assert "Missing required fields" in str(error)
        assert "field2" in error.details["missing_fields"]


class TestProgressIndicator:
    """Test progress indicator context manager."""

    def test_progress_indicator_success(self):
        """Test progress indicator with successful operation."""
        with patch('builtins.print') as mock_print:
            with progress_indicator("test operation"):
                time.sleep(0.01)  # Small delay to test timing

        # Should print start and completion messages
        assert mock_print.call_count >= 2
        calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("Starting test operation" in call for call in calls)
        assert any("completed" in call for call in calls)

    def test_progress_indicator_failure(self):
        """Test progress indicator with failed operation."""
        with patch('builtins.print') as mock_print:
            with pytest.raises(ValueError):
                with progress_indicator("test operation"):
                    raise ValueError("Test error")

        # Should print start and failure messages
        assert mock_print.call_count >= 2
        calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("Starting test operation" in call for call in calls)
        assert any("failed" in call for call in calls)


class TestRetryDecorator:
    """Test retry decorator."""

    def test_with_retry_success_first_attempt(self):
        """Test retry decorator with successful first attempt."""
        @with_retry(max_retries=3, delay=0.01)
        def successful_func():
            return "success"

        result = successful_func()
        assert result == "success"

    def test_with_retry_success_after_retries(self):
        """Test retry decorator with success after retries."""
        attempt_count = 0

        @with_retry(max_retries=3, delay=0.01)
        def retry_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Temporary error")
            return "success"

        with patch('builtins.print'):
            result = retry_func()

        assert result == "success"
        assert attempt_count == 3

    def test_with_retry_all_failures(self):
        """Test retry decorator with all attempts failing."""
        @with_retry(max_retries=2, delay=0.01)
        def failing_func():
            raise ValueError("Persistent error")

        with patch('builtins.print') as mock_print:
            with pytest.raises(ValueError, match="Persistent error"):
                failing_func()

        # Should print retry messages
        calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("Attempt 1 failed" in call for call in calls)
        assert any("Attempt 2 failed" in call for call in calls)


class TestTimeoutDecorator:
    """Test timeout decorator."""

    def test_with_timeout_success(self):
        """Test timeout decorator with successful operation."""
        @with_timeout(1.0)
        def quick_func():
            return "success"

        result = quick_func()
        assert result == "success"

    def test_with_timeout_failure(self):
        """Test timeout decorator with timeout."""
        @with_timeout(0.1)
        def slow_func():
            time.sleep(0.2)
            return "success"

        with pytest.raises(RLDKTimeoutError) as exc_info:
            slow_func()

        error = exc_info.value
        assert "timed out" in str(error)
        assert error.error_code == "OPERATION_TIMEOUT"

    @pytest.mark.skipif(hasattr(signal, 'SIGALRM') is False, reason="SIGALRM not available on this platform")
    def test_with_timeout_signal_handling(self):
        """Test timeout decorator signal handling."""
        @with_timeout(0.1)
        def slow_func():
            time.sleep(0.2)
            return "success"

        # Test that signal is properly restored
        old_handler = signal.signal(signal.SIGALRM, signal.SIG_DFL)

        try:
            with pytest.raises(RLDKTimeoutError):
                slow_func()
        finally:
            # Restore original handler
            signal.signal(signal.SIGALRM, old_handler)


class TestGracefulDegradation:
    """Test graceful degradation decorator."""

    def test_handle_graceful_degradation_success(self):
        """Test graceful degradation with successful operation."""
        @handle_graceful_degradation("test operation", fallback_value="fallback")
        def successful_func():
            return "success"

        result = successful_func()
        assert result == "success"

    def test_handle_graceful_degradation_failure(self):
        """Test graceful degradation with failed operation."""
        @handle_graceful_degradation("test operation", fallback_value="fallback")
        def failing_func():
            raise ValueError("Test error")

        with patch('builtins.print') as mock_print:
            result = failing_func()

        assert result == "fallback"

        # Should print degradation messages
        calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("test operation failed" in call for call in calls)
        assert any("Continuing with degraded functionality" in call for call in calls)


class TestAdapterSourceValidation:
    """Test adapter source validation functions."""

    def test_validate_adapter_source_wandb_success(self):
        """Test WandB URI validation."""
        # Mock the validation function
        with patch('rldk.utils.error_handling.validate_wandb_uri') as mock_validate:
            mock_validate.return_value = {"entity": "test", "project": "test", "run_id": "test"}

            # Should not raise exception
            validate_adapter_source("wandb://entity/project/run_id", ["wandb"])

    def test_validate_adapter_source_wandb_invalid(self):
        """Test WandB URI validation with invalid URI."""
        with patch('rldk.utils.error_handling.validate_wandb_uri') as mock_validate:
            mock_validate.side_effect = ValidationError("Invalid URI")

            with pytest.raises(ValidationError):
                validate_adapter_source("wandb://invalid", ["wandb"])

    def test_validate_adapter_source_file_success(self, temp_dir):
        """Test file source validation."""
        test_file = temp_dir / "test.jsonl"
        test_file.write_text('{"test": "data"}')

        # Should not raise exception
        validate_adapter_source(test_file, ["jsonl"])

    def test_validate_adapter_source_file_not_exists(self, temp_dir):
        """Test file source validation with non-existent file."""
        test_file = temp_dir / "nonexistent.jsonl"

        with pytest.raises(ValidationError) as exc_info:
            validate_adapter_source(test_file, ["jsonl"])

        error = exc_info.value
        assert "does not exist" in str(error)
        assert error.error_code == "SOURCE_NOT_FOUND"

    def test_validate_adapter_source_unsupported_format(self, temp_dir):
        """Test adapter source validation with unsupported format."""
        test_file = temp_dir / "test.unknown"
        test_file.write_text("unknown content")

        with pytest.raises(ValidationError) as exc_info:
            validate_adapter_source(test_file, ["jsonl", "log"])

        error = exc_info.value
        assert "Unsupported source format" in str(error)
        assert error.error_code == "UNSUPPORTED_SOURCE_FORMAT"


class TestUtilityFunctions:
    """Test utility functions."""

    def test_print_usage_examples(self):
        """Test printing usage examples."""
        with patch('builtins.print') as mock_print:
            print_usage_examples("test_command", ["example1", "example2"])

        calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("Usage examples for 'test_command'" in call for call in calls)
        assert any("1. example1" in call for call in calls)
        assert any("2. example2" in call for call in calls)

    def test_print_troubleshooting_tips(self):
        """Test printing troubleshooting tips."""
        with patch('builtins.print') as mock_print:
            print_troubleshooting_tips(["tip1", "tip2"])

        calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("Troubleshooting tips" in call for call in calls)
        assert any("1. tip1" in call for call in calls)
        assert any("2. tip2" in call for call in calls)

    def test_check_dependencies_success(self):
        """Test dependency check with all packages available."""
        with patch('builtins.__import__') as mock_import:
            mock_import.return_value = MagicMock()

            # Should not raise exception
            check_dependencies(["package1", "package2"])

    def test_check_dependencies_missing(self):
        """Test dependency check with missing packages."""
        with patch('builtins.__import__') as mock_import:
            mock_import.side_effect = ImportError("No module named 'missing'")

            with pytest.raises(ValidationError) as exc_info:
                check_dependencies(["missing"])

            error = exc_info.value
            assert "Missing required packages" in str(error)
            assert error.error_code == "MISSING_DEPENDENCIES"

    def test_safe_operation_success(self):
        """Test safe operation decorator with successful operation."""
        @safe_operation("test operation", fallback_value="fallback")
        def successful_func():
            return "success"

        result = successful_func()
        assert result == "success"

    def test_safe_operation_failure(self):
        """Test safe operation decorator with failed operation."""
        @safe_operation("test operation", fallback_value="fallback")
        def failing_func():
            raise ValueError("Test error")

        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            result = failing_func()

        assert result == "fallback"
        mock_logger.warning.assert_called_once()

    def test_safe_operation_no_logging(self):
        """Test safe operation decorator without logging."""
        @safe_operation("test operation", fallback_value="fallback", log_errors=False)
        def failing_func():
            raise ValueError("Test error")

        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            result = failing_func()

        assert result == "fallback"
        mock_logger.warning.assert_not_called()


class TestErrorHandlingIntegration:
    """Test error handling integration scenarios."""

    def test_error_chain(self):
        """Test error chaining with from clause."""
        original_error = ValueError("Original error")

        try:
            raise ValidationError(
                "Validation failed",
                suggestion="Check input",
                error_code="VALIDATION_ERROR"
            ) from original_error
        except ValidationError as e:
            assert isinstance(e, RLDKError)
            assert e.suggestion == "Check input"
            assert e.error_code == "VALIDATION_ERROR"
            assert e.__cause__ == original_error

    def test_error_formatting_with_chain(self):
        """Test error formatting with error chain."""
        original_error = ValueError("Original error")
        try:
            raise ValidationError(
                "Validation failed",
                suggestion="Check input",
                error_code="VALIDATION_ERROR"
            ) from original_error
        except ValidationError as rldk_error:
            message = format_error_message(rldk_error)

            assert "❌ Validation failed" in message
            assert "💡 Suggestion: Check input" in message
            assert "🔍 Error Code: VALIDATION_ERROR" in message

    def test_multiple_decorators(self):
        """Test combining multiple decorators."""
        @with_retry(max_retries=2, delay=0.01)
        @with_timeout(1.0)
        @safe_operation("test operation", fallback_value="fallback")
        def complex_func():
            return "success"

        result = complex_func()
        assert result == "success"

    def test_error_handling_in_context_manager(self):
        """Test error handling within context manager."""
        with pytest.raises(ValueError):
            with progress_indicator("test operation"):
                raise ValueError("Test error")

        # Context manager should handle the error and re-raise it


if __name__ == "__main__":
    pytest.main([__file__])
