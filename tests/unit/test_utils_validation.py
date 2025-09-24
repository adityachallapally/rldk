#!/usr/bin/env python3
"""Unit tests for rldk.utils.validation module."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import numpy as np
import pandas as pd
import pytest

from rldk.utils.error_handling import ValidationError

# Import the module under test
from rldk.utils.validation import (
    validate_adapter_type,
    validate_choice,
    validate_data_quality,
    validate_dataframe,
    validate_directory_exists,
    validate_evaluation_suite,
    validate_file_exists,
    validate_file_extension,
    validate_file_size,
    validate_json_file,
    validate_jsonl_file,
    validate_non_negative_integer,
    validate_numeric_range,
    validate_optional_positive_integer,
    validate_optional_string,
    validate_positive_integer,
    validate_probability,
    validate_string_not_empty,
    validate_wandb_uri,
    validate_with_custom_validator,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    rows = 100
    return pd.DataFrame({
        "numbers": np.arange(rows),
        "strings": [f"s{i}" for i in range(rows)],
        "mixed": np.where(np.arange(rows) % 2 == 0, np.arange(rows), np.arange(rows) * 0.1),
        "step": np.arange(rows),
        "reward_mean": np.random.normal(0.5, 0.2, rows),
        "reward_std": np.random.uniform(0.1, 0.3, rows),
        "tokens_out": np.random.randint(10, 100, rows),
        "repetition_penalty": np.random.uniform(0.8, 1.2, rows),
        "human_preference": np.random.uniform(0, 1, rows),
        "ground_truth": np.random.choice([0, 1], rows),
        "epoch": np.random.randint(0, 10, rows),
        "run_id": ["test_run"] * rows,
    })


class TestFileValidation:
    """Test file validation functions."""

    def test_validate_file_exists_success(self, temp_dir):
        """Test successful file validation."""
        # Create a test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        # Should return the path
        result = validate_file_exists(test_file)
        assert result == test_file
        assert isinstance(result, Path)

    def test_validate_file_exists_not_found(self, temp_dir):
        """Test file validation when file doesn't exist."""
        test_file = temp_dir / "nonexistent.txt"

        with pytest.raises(ValidationError) as exc_info:
            validate_file_exists(test_file)

        error = exc_info.value
        assert "does not exist" in str(error)
        assert error.error_code == "FILE_NOT_FOUND"

    def test_validate_file_exists_not_a_file(self, temp_dir):
        """Test file validation when path is a directory."""
        # Create a directory
        test_dir = temp_dir / "test_dir"
        test_dir.mkdir()

        with pytest.raises(ValidationError) as exc_info:
            validate_file_exists(test_dir)

        error = exc_info.value
        assert "not a file" in str(error)
        assert error.error_code == "NOT_A_FILE"

    def test_validate_file_exists_no_permission(self, temp_dir):
        """Test file validation when no read permission."""
        # Create a test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        # Mock os.access to return False
        with patch('os.access', return_value=False):
            with pytest.raises(ValidationError) as exc_info:
                validate_file_exists(test_file)

            error = exc_info.value
            assert "No read permission" in str(error)
            assert error.error_code == "PERMISSION_DENIED"

    def test_validate_directory_exists_success(self, temp_dir):
        """Test successful directory validation."""
        # Create a test directory
        test_dir = temp_dir / "test_dir"
        test_dir.mkdir()

        # Should return the path
        result = validate_directory_exists(test_dir)
        assert result == test_dir
        assert isinstance(result, Path)

    def test_validate_directory_exists_not_found(self, temp_dir):
        """Test directory validation when directory doesn't exist."""
        test_dir = temp_dir / "nonexistent_dir"

        with pytest.raises(ValidationError) as exc_info:
            validate_directory_exists(test_dir)

        error = exc_info.value
        assert "does not exist" in str(error)
        assert error.error_code == "DIRECTORY_NOT_FOUND"

    def test_validate_directory_exists_not_a_directory(self, temp_dir):
        """Test directory validation when path is a file."""
        # Create a test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        with pytest.raises(ValidationError) as exc_info:
            validate_directory_exists(test_file)

        error = exc_info.value
        assert "not a directory" in str(error)
        assert error.error_code == "NOT_A_DIRECTORY"

    def test_validate_file_extension_success(self, temp_dir):
        """Test successful file extension validation."""
        # Create a test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        # Should return the path
        result = validate_file_extension(test_file, [".txt", ".json"])
        assert result == test_file

    def test_validate_file_extension_case_insensitive(self, temp_dir):
        """Test file extension validation is case insensitive."""
        # Create a test file
        test_file = temp_dir / "test.TXT"
        test_file.write_text("test content")

        # Should work with lowercase extension
        result = validate_file_extension(test_file, [".txt"])
        assert result == test_file

    def test_validate_file_extension_invalid(self, temp_dir):
        """Test file extension validation with invalid extension."""
        # Create a test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        with pytest.raises(ValidationError) as exc_info:
            validate_file_extension(test_file, [".json", ".csv"])

        error = exc_info.value
        assert "unsupported extension" in str(error)
        assert error.error_code == "UNSUPPORTED_EXTENSION"


class TestJSONValidation:
    """Test JSON validation functions."""

    def test_validate_json_file_success(self, temp_dir):
        """Test successful JSON file validation."""
        # Create a test JSON file
        test_file = temp_dir / "test.json"
        test_data = {"key": "value", "number": 42}
        test_file.write_text(json.dumps(test_data))

        # Should return the parsed data
        result = validate_json_file(test_file)
        assert result == test_data
        assert isinstance(result, dict)

    def test_validate_json_file_invalid_json(self, temp_dir):
        """Test JSON file validation with invalid JSON."""
        # Create a test file with invalid JSON
        test_file = temp_dir / "test.json"
        test_file.write_text("{ invalid json }")

        with pytest.raises(ValidationError) as exc_info:
            validate_json_file(test_file)

        error = exc_info.value
        assert "Invalid JSON" in str(error)
        assert error.error_code == "INVALID_JSON"

    def test_validate_json_file_not_dict(self, temp_dir):
        """Test JSON file validation when JSON is not a dictionary."""
        # Create a test file with JSON array
        test_file = temp_dir / "test.json"
        test_file.write_text('[1, 2, 3]')

        with pytest.raises(ValidationError) as exc_info:
            validate_json_file(test_file)

        error = exc_info.value
        assert "not contain a dictionary" in str(error)
        assert error.error_code == "INVALID_JSON_STRUCTURE"

    def test_validate_jsonl_file_success(self, temp_dir):
        """Test successful JSONL file validation."""
        # Create a test JSONL file
        test_file = temp_dir / "test.jsonl"
        test_data = [
            {"key1": "value1"},
            {"key2": "value2"},
            {"key3": "value3"}
        ]
        test_file.write_text('\n'.join(json.dumps(record) for record in test_data))

        # Should return the parsed data
        result = validate_jsonl_file(test_file)
        assert result == test_data
        assert isinstance(result, list)
        assert len(result) == 3

    def test_validate_jsonl_file_empty(self, temp_dir):
        """Test JSONL file validation with empty file."""
        # Create an empty test file
        test_file = temp_dir / "test.jsonl"
        test_file.write_text("")

        with pytest.raises(ValidationError) as exc_info:
            validate_jsonl_file(test_file)

        error = exc_info.value
        assert "No valid JSON records" in str(error)
        assert error.error_code == "EMPTY_JSONL_FILE"

    def test_validate_jsonl_file_invalid_line(self, temp_dir):
        """Test JSONL file validation with invalid line."""
        # Create a test file with invalid JSON line
        test_file = temp_dir / "test.jsonl"
        test_file.write_text('{"valid": "json"}\n{ invalid json }\n{"another": "valid"}')

        with pytest.raises(ValidationError) as exc_info:
            validate_jsonl_file(test_file)

        error = exc_info.value
        assert "Invalid JSON on line 2" in str(error)
        assert error.error_code == "INVALID_JSONL_LINE"


class TestDataFrameValidation:
    """Test DataFrame validation functions."""

    def test_validate_dataframe_success(self, sample_data):
        """Test successful DataFrame validation."""
        result = validate_dataframe(sample_data)
        assert result is sample_data

        # Test with required columns
        result = validate_dataframe(sample_data, required_columns=["numbers", "strings"])
        assert result is sample_data

    def test_validate_dataframe_invalid_type(self):
        """Test DataFrame validation with invalid type."""
        with pytest.raises(ValidationError) as exc_info:
            validate_dataframe("not a dataframe")

        error = exc_info.value
        assert "must be a pandas DataFrame" in str(error)
        assert error.error_code == "INVALID_DATAFRAME_TYPE"

    def test_validate_dataframe_insufficient_rows(self):
        """Test DataFrame validation with insufficient rows."""
        df = pd.DataFrame({"col1": [1, 2]})

        with pytest.raises(ValidationError) as exc_info:
            validate_dataframe(df, min_rows=5)

        error = exc_info.value
        assert "too few rows" in str(error)
        assert error.error_code == "INSUFFICIENT_ROWS"

    def test_validate_dataframe_missing_columns(self, sample_data):
        """Test DataFrame validation with missing columns."""
        with pytest.raises(ValidationError) as exc_info:
            validate_dataframe(sample_data, required_columns=["missing_col"])

        error = exc_info.value
        assert "missing required columns" in str(error)
        assert error.error_code == "MISSING_COLUMNS"

    def test_validate_data_quality_success(self, sample_data):
        """Test successful data quality validation."""
        result = validate_data_quality(sample_data, ["step", "reward_mean"])
        assert result is sample_data

    def test_validate_data_quality_poor_quality(self):
        """Test data quality validation with poor quality data."""
        # Create DataFrame with many missing values
        df = pd.DataFrame({
            "col1": [1, 2, None, None, None],
            "col2": [1, 2, 3, 4, 5]
        })

        with pytest.raises(ValidationError) as exc_info:
            validate_data_quality(df, ["col1"], max_missing_ratio=0.3)

        error = exc_info.value
        assert "too many missing values" in str(error)
        assert error.error_code == "POOR_DATA_QUALITY"


class TestNumericValidation:
    """Test numeric validation functions."""

    def test_validate_numeric_range_success(self):
        """Test successful numeric range validation."""
        # Test within range
        result = validate_numeric_range(5, 0, 10)
        assert result == 5

        # Test at boundaries
        result = validate_numeric_range(0, 0, 10)
        assert result == 0

        result = validate_numeric_range(10, 0, 10)
        assert result == 10

    def test_validate_numeric_range_invalid_type(self):
        """Test numeric range validation with invalid type."""
        with pytest.raises(ValidationError) as exc_info:
            validate_numeric_range("not a number")

        error = exc_info.value
        assert "must be numeric" in str(error)
        assert error.error_code == "INVALID_NUMERIC_TYPE"

    def test_validate_numeric_range_non_finite(self):
        """Test numeric range validation with non-finite value."""
        with pytest.raises(ValidationError) as exc_info:
            validate_numeric_range(float('inf'))

        error = exc_info.value
        assert "must be finite" in str(error)
        assert error.error_code == "NON_FINITE_VALUE"

    def test_validate_numeric_range_too_small(self):
        """Test numeric range validation with value too small."""
        with pytest.raises(ValidationError) as exc_info:
            validate_numeric_range(5, 10, 20)

        error = exc_info.value
        assert "must be >=" in str(error)
        assert error.error_code == "VALUE_TOO_SMALL"

    def test_validate_numeric_range_too_large(self):
        """Test numeric range validation with value too large."""
        with pytest.raises(ValidationError) as exc_info:
            validate_numeric_range(25, 10, 20)

        error = exc_info.value
        assert "must be <=" in str(error)
        assert error.error_code == "VALUE_TOO_LARGE"

    def test_validate_positive_integer_success(self):
        """Test successful positive integer validation."""
        result = validate_positive_integer(5)
        assert result == 5

        # Test with string
        result = validate_positive_integer("5")
        assert result == 5

        # Test with float that is integer
        result = validate_positive_integer(5.0)
        assert result == 5

    def test_validate_positive_integer_invalid_type(self):
        """Test positive integer validation with invalid type."""
        with pytest.raises(ValidationError) as exc_info:
            validate_positive_integer("not a number")

        error = exc_info.value
        assert "must be a number" in str(error)
        assert error.error_code == "INVALID_NUMERIC_TYPE"

    def test_validate_positive_integer_non_integer(self):
        """Test positive integer validation with non-integer."""
        with pytest.raises(ValidationError) as exc_info:
            validate_positive_integer(5.5)

        error = exc_info.value
        assert "must be an integer" in str(error)
        assert error.error_code == "NON_INTEGER_VALUE"

    def test_validate_positive_integer_non_positive(self):
        """Test positive integer validation with non-positive value."""
        with pytest.raises(ValidationError) as exc_info:
            validate_positive_integer(0)

        error = exc_info.value
        assert "must be positive" in str(error)
        assert error.error_code == "NON_POSITIVE_VALUE"

    def test_validate_non_negative_integer_success(self):
        """Test successful non-negative integer validation."""
        result = validate_non_negative_integer(0)
        assert result == 0

        result = validate_non_negative_integer(5)
        assert result == 5

    def test_validate_non_negative_integer_negative(self):
        """Test non-negative integer validation with negative value."""
        with pytest.raises(ValidationError) as exc_info:
            validate_non_negative_integer(-1)

        error = exc_info.value
        assert "must be non-negative" in str(error)
        assert error.error_code == "NEGATIVE_VALUE"

    def test_validate_probability_success(self):
        """Test successful probability validation."""
        result = validate_probability(0.5)
        assert result == 0.5

        result = validate_probability(0.0)
        assert result == 0.0

        result = validate_probability(1.0)
        assert result == 1.0

    def test_validate_probability_out_of_range(self):
        """Test probability validation with out of range value."""
        with pytest.raises(ValidationError) as exc_info:
            validate_probability(1.5)

        error = exc_info.value
        assert "must be <=" in str(error)
        assert error.error_code == "VALUE_TOO_LARGE"


class TestStringValidation:
    """Test string validation functions."""

    def test_validate_string_not_empty_success(self):
        """Test successful string validation."""
        result = validate_string_not_empty("hello")
        assert result == "hello"

        # Test with whitespace
        result = validate_string_not_empty("  hello  ")
        assert result == "hello"

    def test_validate_string_not_empty_invalid_type(self):
        """Test string validation with invalid type."""
        with pytest.raises(ValidationError) as exc_info:
            validate_string_not_empty(123)

        error = exc_info.value
        assert "must be a string" in str(error)
        assert error.error_code == "INVALID_STRING_TYPE"

    def test_validate_string_not_empty_empty(self):
        """Test string validation with empty string."""
        with pytest.raises(ValidationError) as exc_info:
            validate_string_not_empty("")

        error = exc_info.value
        assert "cannot be empty" in str(error)
        assert error.error_code == "EMPTY_STRING"

    def test_validate_string_not_empty_whitespace_only(self):
        """Test string validation with whitespace-only string."""
        with pytest.raises(ValidationError) as exc_info:
            validate_string_not_empty("   ")

        error = exc_info.value
        assert "cannot be empty" in str(error)
        assert error.error_code == "EMPTY_STRING"

    def test_validate_optional_string_success(self):
        """Test successful optional string validation."""
        result = validate_optional_string("hello")
        assert result == "hello"

        result = validate_optional_string(None)
        assert result is None

    def test_validate_optional_string_empty(self):
        """Test optional string validation with empty string."""
        with pytest.raises(ValidationError) as exc_info:
            validate_optional_string("")

        error = exc_info.value
        assert "cannot be empty" in str(error)
        assert error.error_code == "EMPTY_STRING"


class TestChoiceValidation:
    """Test choice validation functions."""

    def test_validate_choice_success(self):
        """Test successful choice validation."""
        result = validate_choice("option1", ["option1", "option2", "option3"])
        assert result == "option1"

    def test_validate_choice_invalid(self):
        """Test choice validation with invalid choice."""
        with pytest.raises(ValidationError) as exc_info:
            validate_choice("invalid", ["option1", "option2"])

        error = exc_info.value
        assert "must be one of" in str(error)
        assert error.error_code == "INVALID_CHOICE"

    def test_validate_adapter_type_success(self):
        """Test successful adapter type validation."""
        result = validate_adapter_type("trl")
        assert result == "trl"

        result = validate_adapter_type("openrlhf")
        assert result == "openrlhf"

    def test_validate_adapter_type_invalid(self):
        """Test adapter type validation with invalid type."""
        with pytest.raises(ValidationError) as exc_info:
            validate_adapter_type("invalid")

        error = exc_info.value
        assert "must be one of" in str(error)
        assert error.error_code == "INVALID_CHOICE"

    def test_validate_evaluation_suite_success(self):
        """Test successful evaluation suite validation."""
        result = validate_evaluation_suite("quick")
        assert result == "quick"

        result = validate_evaluation_suite("comprehensive")
        assert result == "comprehensive"

    def test_validate_evaluation_suite_invalid(self):
        """Test evaluation suite validation with invalid suite."""
        with pytest.raises(ValidationError) as exc_info:
            validate_evaluation_suite("invalid")

        error = exc_info.value
        assert "must be one of" in str(error)
        assert error.error_code == "INVALID_CHOICE"


class TestWandBValidation:
    """Test WandB URI validation functions."""

    def test_validate_wandb_uri_success(self):
        """Test successful WandB URI validation."""
        uri = "wandb://entity/project/run_id"
        result = validate_wandb_uri(uri)

        expected = {
            "entity": "entity",
            "project": "project",
            "run_id": "run_id"
        }
        assert result == expected

    def test_validate_wandb_uri_invalid_type(self):
        """Test WandB URI validation with invalid type."""
        with pytest.raises(ValidationError) as exc_info:
            validate_wandb_uri(123)

        error = exc_info.value
        assert "must be a string" in str(error)
        assert error.error_code == "INVALID_URI_TYPE"

    def test_validate_wandb_uri_invalid_prefix(self):
        """Test WandB URI validation with invalid prefix."""
        with pytest.raises(ValidationError) as exc_info:
            validate_wandb_uri("http://entity/project/run_id")

        error = exc_info.value
        assert "must start with 'wandb://'" in str(error)
        assert error.error_code == "INVALID_URI_PREFIX"

    def test_validate_wandb_uri_invalid_format(self):
        """Test WandB URI validation with invalid format."""
        with pytest.raises(ValidationError) as exc_info:
            validate_wandb_uri("wandb://entity/project")

        error = exc_info.value
        assert "must have 3 parts" in str(error)
        assert error.error_code == "INVALID_URI_FORMAT"

    def test_validate_wandb_uri_empty_parts(self):
        """Test WandB URI validation with empty parts."""
        with pytest.raises(ValidationError) as exc_info:
            validate_wandb_uri("wandb://entity//run_id")

        error = exc_info.value
        assert "project cannot be empty" in str(error)
        assert error.error_code == "EMPTY_PROJECT"


class TestFileSizeValidation:
    """Test file size validation functions."""

    def test_validate_file_size_success(self, temp_dir):
        """Test successful file size validation."""
        # Create a small test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("small content")

        result = validate_file_size(test_file, max_size_mb=1.0)
        assert result == test_file

    def test_validate_file_size_too_large(self, temp_dir):
        """Test file size validation with file too large."""
        # Create a test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("small content")

        # Mock file size to be large
        with patch.object(Path, 'stat') as mock_stat:
            mock_stat.return_value.st_size = 2 * 1024 * 1024  # 2MB

            with pytest.raises(ValidationError) as exc_info:
                validate_file_size(test_file, max_size_mb=1.0)

            error = exc_info.value
            assert "too large" in str(error)
            assert error.error_code == "FILE_TOO_LARGE"


class TestCustomValidation:
    """Test custom validation functions."""

    def test_validate_with_custom_validator_success(self):
        """Test successful custom validation."""
        def is_even(value):
            return value % 2 == 0

        result = validate_with_custom_validator(4, is_even, "must be even")
        assert result == 4

    def test_validate_with_custom_validator_failure(self):
        """Test custom validation failure."""
        def is_even(value):
            return value % 2 == 0

        with pytest.raises(ValidationError) as exc_info:
            validate_with_custom_validator(3, is_even, "must be even")

        error = exc_info.value
        assert "validation failed" in str(error)
        assert error.error_code == "CUSTOM_VALIDATION_FAILED"

    def test_validate_with_custom_validator_exception(self):
        """Test custom validation with validator exception."""
        def failing_validator(value):
            raise ValueError("validator error")

        with pytest.raises(ValidationError) as exc_info:
            validate_with_custom_validator(3, failing_validator, "must be even")

        error = exc_info.value
        assert "Custom validation error" in str(error)
        assert error.error_code == "VALIDATOR_ERROR"


class TestOptionalValidation:
    """Test optional validation functions."""

    def test_validate_optional_positive_integer_success(self):
        """Test successful optional positive integer validation."""
        result = validate_optional_positive_integer(5)
        assert result == 5

        result = validate_optional_positive_integer(None)
        assert result is None

    def test_validate_optional_positive_integer_invalid(self):
        """Test optional positive integer validation with invalid value."""
        with pytest.raises(ValidationError) as exc_info:
            validate_optional_positive_integer(-1)

        error = exc_info.value
        assert "must be positive" in str(error)
        assert error.error_code == "NON_POSITIVE_VALUE"


if __name__ == "__main__":
    pytest.main([__file__])
