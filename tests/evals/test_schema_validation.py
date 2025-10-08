"""Tests for schema validation and data normalization."""

import numpy as np
import pandas as pd
import pytest

from rldk.evals.schema import (
    STANDARD_EVAL_SCHEMA,
    ColumnSpec,
    EvalInputSchema,
    get_schema_for_suite,
    safe_mean,
    validate_eval_input,
)


class TestColumnSpec:
    """Test ColumnSpec class."""

    def test_column_spec_creation(self):
        """Test creating a ColumnSpec."""
        spec = ColumnSpec(
            name="test_column",
            dtype="numeric",
            required=True,
            description="Test column",
            example=42,
            synonyms=["test_col", "test"]
        )

        assert spec.name == "test_column"
        assert spec.dtype == "numeric"
        assert spec.required is True
        assert spec.description == "Test column"
        assert spec.example == 42
        assert spec.synonyms == ["test_col", "test"]

    def test_column_spec_default_synonyms(self):
        """Test ColumnSpec with default synonyms."""
        spec = ColumnSpec(
            name="test_column",
            dtype="numeric",
            required=True,
            description="Test column",
            example=42
        )

        assert spec.synonyms == []


class TestEvalInputSchema:
    """Test EvalInputSchema class."""

    def test_schema_creation(self):
        """Test creating an EvalInputSchema."""
        required = [
            ColumnSpec("col1", "numeric", True, "Required column", 1)
        ]
        optional = [
            ColumnSpec("col2", "text", False, "Optional column", "test")
        ]

        schema = EvalInputSchema(required, optional)

        assert len(schema.required_columns) == 1
        assert len(schema.optional_columns) == 1
        assert len(schema.get_all_columns()) == 2

    def test_get_column_by_name(self):
        """Test getting column by name or synonym."""
        required = [
            ColumnSpec("step", "numeric", True, "Step column", 1, ["global_step"])
        ]
        schema = EvalInputSchema(required, [])

        # Test exact match
        col = schema.get_column_by_name("step")
        assert col is not None
        assert col.name == "step"

        # Test synonym match
        col = schema.get_column_by_name("global_step")
        assert col is not None
        assert col.name == "step"

        # Test no match
        col = schema.get_column_by_name("nonexistent")
        assert col is None

    def test_get_column_names(self):
        """Test getting all column names and synonyms."""
        required = [
            ColumnSpec("step", "numeric", True, "Step column", 1, ["global_step"]),
            ColumnSpec("output", "text", True, "Output column", "test", ["response"])
        ]
        schema = EvalInputSchema(required, [])

        names = schema.get_column_names()
        expected = ["step", "global_step", "output", "response"]
        assert set(names) == set(expected)


class TestValidateEvalInput:
    """Test validate_eval_input function."""

    def test_successful_validation(self):
        """Test successful validation with all required columns."""
        data = pd.DataFrame({
            'step': [1, 2, 3],
            'output': ['response1', 'response2', 'response3']
        })

        result = validate_eval_input(data, STANDARD_EVAL_SCHEMA, "test")

        assert len(result.warnings) == 1  # Missing events warning
        assert len(result.errors) == 0
        assert 'step' in result.data.columns
        assert 'output' in result.data.columns

    def test_column_normalization(self):
        """Test automatic column normalization."""
        data = pd.DataFrame({
            'global_step': [1, 2, 3],  # Should normalize to 'step'
            'response': ['r1', 'r2', 'r3']  # Should normalize to 'output'
        })

        result = validate_eval_input(data, STANDARD_EVAL_SCHEMA, "test")

        assert 'step' in result.data.columns
        assert 'output' in result.data.columns
        assert 'global_step' not in result.data.columns
        assert 'response' not in result.data.columns
        assert result.normalized_columns['global_step'] == 'step'
        assert result.normalized_columns['response'] == 'output'

    def test_missing_required_column(self):
        """Test validation failure with missing required column."""
        data = pd.DataFrame({
            'step': [1, 2, 3]
            # Missing 'output' column
        })

        with pytest.raises(ValueError) as exc_info:
            validate_eval_input(data, STANDARD_EVAL_SCHEMA, "test")

        assert "Missing required columns: output" in str(exc_info.value)
        assert "Provide one of: output, response, completion, text" in str(exc_info.value)

    def test_missing_step_column(self):
        """Test validation failure with missing step column."""
        data = pd.DataFrame({
            'output': ['r1', 'r2', 'r3']
            # Missing 'step' column
        })

        with pytest.raises(ValueError) as exc_info:
            validate_eval_input(data, STANDARD_EVAL_SCHEMA, "test")

        assert "Missing required columns: step" in str(exc_info.value)
        assert "Provide one of: step, global_step, iteration, epoch" in str(exc_info.value)

    def test_missing_optional_columns(self):
        """Test validation with missing optional columns."""
        data = pd.DataFrame({
            'step': [1, 2, 3],
            'output': ['r1', 'r2', 'r3']
            # Missing optional columns: reward, kl_to_ref, events
        })

        result = validate_eval_input(data, STANDARD_EVAL_SCHEMA, "test")

        assert len(result.warnings) == 1  # Only events warning
        assert "events column not provided" in result.warnings[0]
        assert len(result.errors) == 0

    def test_empty_dataframe(self):
        """Test validation with empty DataFrame."""
        data = pd.DataFrame({
            'step': [],
            'output': []
        })

        result = validate_eval_input(data, STANDARD_EVAL_SCHEMA, "test")

        assert len(result.warnings) >= 1
        assert "DataFrame is empty" in result.warnings

    def test_all_nan_column(self):
        """Test validation with all-NaN column."""
        data = pd.DataFrame({
            'step': [1, 2, 3],
            'output': ['r1', 'r2', 'r3'],
            'reward': [np.nan, np.nan, np.nan]
        })

        result = validate_eval_input(data, STANDARD_EVAL_SCHEMA, "test")

        assert len(result.warnings) >= 1
        assert any("contains only NaN values" in w for w in result.warnings)

    def test_dtype_validation(self):
        """Test dtype validation."""
        data = pd.DataFrame({
            'step': [1, 2, 3],
            'output': ['r1', 'r2', 'r3'],
            'reward': ['not', 'numeric', 'data']  # Should be numeric
        })

        result = validate_eval_input(data, STANDARD_EVAL_SCHEMA, "test")

        # Should have warnings about dtype issues
        assert len(result.warnings) >= 1
        assert any("may not be valid numeric" in w for w in result.warnings)


class TestSafeMean:
    """Test safe_mean function."""

    def test_safe_mean_with_values(self):
        """Test safe_mean with valid values."""
        values = [1.0, 2.0, 3.0, 4.0]
        result = safe_mean(values)
        assert result == 2.5

    def test_safe_mean_empty_list(self):
        """Test safe_mean with empty list."""
        result = safe_mean([])
        assert result is None

    def test_safe_mean_with_nan(self):
        """Test safe_mean with NaN values."""
        values = [1.0, np.nan, 3.0, np.nan]
        result = safe_mean(values)
        assert result == 2.0

    def test_safe_mean_all_nan(self):
        """Test safe_mean with all NaN values."""
        values = [np.nan, np.nan, np.nan]
        result = safe_mean(values)
        assert result is None

    def test_safe_mean_mixed_types(self):
        """Test safe_mean with mixed numeric types."""
        values = [1, 2.0, 3, 4.5]
        result = safe_mean(values)
        assert result == 2.625


class TestStandardSchema:
    """Test STANDARD_EVAL_SCHEMA."""

    def test_standard_schema_structure(self):
        """Test that standard schema has expected structure."""
        schema = STANDARD_EVAL_SCHEMA

        # Check required columns
        required_names = [col.name for col in schema.required_columns]
        assert 'step' in required_names
        assert 'output' in required_names

        # Check optional columns
        optional_names = [col.name for col in schema.optional_columns]
        assert 'reward' in optional_names
        assert 'kl_to_ref' in optional_names
        assert 'events' in optional_names

    def test_step_column_synonyms(self):
        """Test step column synonyms."""
        schema = STANDARD_EVAL_SCHEMA
        step_col = schema.get_column_by_name("step")

        assert step_col is not None
        assert 'global_step' in step_col.synonyms
        assert 'iteration' in step_col.synonyms
        assert 'epoch' in step_col.synonyms

    def test_output_column_synonyms(self):
        """Test output column synonyms."""
        schema = STANDARD_EVAL_SCHEMA
        output_col = schema.get_column_by_name("output")

        assert output_col is not None
        assert 'response' in output_col.synonyms
        assert 'completion' in output_col.synonyms
        assert 'text' in output_col.synonyms
        assert 'generation' in output_col.synonyms


class TestGetSchemaForSuite:
    """Test get_schema_for_suite function."""

    def test_get_schema_for_suite(self):
        """Test getting schema for different suites."""
        # All suites should return the standard schema for now
        schema1 = get_schema_for_suite("quick")
        schema2 = get_schema_for_suite("comprehensive")
        schema3 = get_schema_for_suite("safety")

        assert schema1 == STANDARD_EVAL_SCHEMA
        assert schema2 == STANDARD_EVAL_SCHEMA
        assert schema3 == STANDARD_EVAL_SCHEMA


class TestIntegration:
    """Integration tests for schema validation."""

    def test_full_validation_workflow(self):
        """Test complete validation workflow."""
        # Create data with synonyms
        data = pd.DataFrame({
            'global_step': [1, 2, 3, 4, 5],
            'response': ['r1', 'r2', 'r3', 'r4', 'r5'],
            'reward_mean': [0.8, 0.9, 0.7, 0.85, 0.95],
            'kl': [0.1, 0.12, 0.08, 0.11, 0.09]
        })

        # Validate input
        result = validate_eval_input(data, STANDARD_EVAL_SCHEMA, "test")

        # Check normalization
        assert 'step' in result.data.columns
        assert 'output' in result.data.columns
        assert 'reward' in result.data.columns
        assert 'kl_to_ref' in result.data.columns

        # Check warnings
        assert len(result.warnings) == 1  # Only events warning
        assert "events column not provided" in result.warnings[0]

        # Check errors
        assert len(result.errors) == 0
