"""Tests for EvalResult contract and attribute consistency."""

import numpy as np
import pandas as pd
import pytest

from rldk.evals.runner import EvalResult


class TestEvalResultContract:
    """Test EvalResult contract and required attributes."""

    def test_eval_result_has_required_attributes(self):
        """Test that EvalResult has all required attributes."""
        result = EvalResult(
            suite_name="test",
            scores={"metric1": 0.8, "metric2": 0.6},
            confidence_intervals={"metric1": (0.7, 0.9), "metric2": (0.5, 0.7)},
            effect_sizes={"metric1": 0.5, "metric2": 0.3},
            sample_size=100,
            seed=42,
            metadata={"test": "value"},
            raw_results=[{"evaluation": "metric1", "result": {"score": 0.8}}],
            warnings=["test warning"]
        )

        # Test required attributes
        assert hasattr(result, 'suite_name')
        assert hasattr(result, 'scores')
        assert hasattr(result, 'confidence_intervals')
        assert hasattr(result, 'effect_sizes')
        assert hasattr(result, 'sample_size')
        assert hasattr(result, 'seed')
        assert hasattr(result, 'metadata')
        assert hasattr(result, 'raw_results')
        assert hasattr(result, 'warnings')

        # Test property attributes
        assert hasattr(result, 'overall_score')
        assert hasattr(result, 'available_fraction')

    def test_eval_result_default_warnings(self):
        """Test that EvalResult initializes warnings as empty list by default."""
        result = EvalResult(
            suite_name="test",
            scores={},
            confidence_intervals={},
            effect_sizes={},
            sample_size=0,
            seed=42,
            metadata={},
            raw_results=[]
        )

        assert result.warnings == []

    def test_eval_result_custom_warnings(self):
        """Test that EvalResult accepts custom warnings."""
        custom_warnings = ["warning1", "warning2"]
        result = EvalResult(
            suite_name="test",
            scores={},
            confidence_intervals={},
            effect_sizes={},
            sample_size=0,
            seed=42,
            metadata={},
            raw_results=[],
            warnings=custom_warnings
        )

        assert result.warnings == custom_warnings


class TestOverallScoreProperty:
    """Test overall_score property behavior."""

    def test_overall_score_is_property(self):
        """Test that overall_score is a property, not a field."""
        result = EvalResult(
            suite_name="test",
            scores={"metric1": 0.8, "metric2": 0.6},
            confidence_intervals={},
            effect_sizes={},
            sample_size=100,
            seed=42,
            metadata={},
            raw_results=[]
        )

        # Should be a property, not a field
        assert isinstance(type(result).overall_score, property)

        # Should be callable
        score = result.overall_score
        assert score == 0.7  # (0.8 + 0.6) / 2

    def test_overall_score_computation(self):
        """Test overall_score computation logic."""
        test_cases = [
            # (scores, expected_overall_score)
            ({"a": 1.0, "b": 2.0}, 1.5),  # Simple average
            ({"a": 0.5, "b": 0.7, "c": 0.9}, 0.7),  # Three metrics
            ({"a": 0.8, "b": np.nan, "c": 0.6}, 0.7),  # One NaN
            ({"a": 0.8, "b": None, "c": 0.6}, 0.7),  # One None
            ({"a": np.nan, "b": np.nan}, None),  # All NaN
            ({"a": None, "b": None}, None),  # All None
            ({}, None),  # Empty scores
        ]

        for scores, expected in test_cases:
            result = EvalResult(
                suite_name="test",
                scores=scores,
                confidence_intervals={},
                effect_sizes={},
                sample_size=100,
                seed=42,
                metadata={},
                raw_results=[]
            )

            if expected is None:
                assert result.overall_score is None
            else:
                assert abs(result.overall_score - expected) < 1e-10, \
                    f"Expected {expected}, got {result.overall_score} for scores {scores}"

    def test_overall_score_ignores_invalid_values(self):
        """Test that overall_score ignores NaN and None values."""
        scores = {
            "valid1": 0.8,
            "nan_metric": np.nan,
            "valid2": 0.6,
            "none_metric": None,
            "valid3": 0.4
        }

        result = EvalResult(
            suite_name="test",
            scores=scores,
            confidence_intervals={},
            effect_sizes={},
            sample_size=100,
            seed=42,
            metadata={},
            raw_results=[]
        )

        # Should only consider valid1, valid2, valid3
        expected = (0.8 + 0.6 + 0.4) / 3
        assert abs(result.overall_score - expected) < 1e-10


class TestAvailableFractionProperty:
    """Test available_fraction property behavior."""

    def test_available_fraction_is_property(self):
        """Test that available_fraction is a property, not a field."""
        result = EvalResult(
            suite_name="test",
            scores={"metric1": 0.8, "metric2": 0.6},
            confidence_intervals={},
            effect_sizes={},
            sample_size=100,
            seed=42,
            metadata={},
            raw_results=[]
        )

        # Should be a property, not a field
        assert isinstance(type(result).available_fraction, property)

        # Should be callable
        fraction = result.available_fraction
        assert fraction == 1.0

    def test_available_fraction_calculation(self):
        """Test available_fraction calculation logic."""
        test_cases = [
            # (scores, expected_fraction)
            ({"a": 1.0, "b": 2.0}, 1.0),  # All valid
            ({"a": 0.8, "b": np.nan}, 0.5),  # Half valid
            ({"a": 0.8, "b": None}, 0.5),  # Half valid
            ({"a": np.nan, "b": np.nan}, 0.0),  # None valid
            ({"a": None, "b": None}, 0.0),  # None valid
            ({"a": 0.8, "b": np.nan, "c": 0.6}, 2/3),  # Two-thirds valid
            ({}, 0.0),  # Empty scores
        ]

        for scores, expected in test_cases:
            result = EvalResult(
                suite_name="test",
                scores=scores,
                confidence_intervals={},
                effect_sizes={},
                sample_size=100,
                seed=42,
                metadata={},
                raw_results=[]
            )

            assert abs(result.available_fraction - expected) < 1e-10, \
                f"Expected {expected}, got {result.available_fraction} for scores {scores}"

    def test_available_fraction_range(self):
        """Test that available_fraction is always in [0, 1] range."""
        test_scores = [
            {"a": 0.8, "b": 0.6, "c": 0.7},  # All valid
            {"a": 0.8, "b": np.nan, "c": 0.7},  # Some valid
            {"a": np.nan, "b": np.nan, "c": np.nan},  # None valid
            {"a": 0.8, "b": None, "c": 0.7, "d": np.nan},  # Mixed
        ]

        for scores in test_scores:
            result = EvalResult(
                suite_name="test",
                scores=scores,
                confidence_intervals={},
                effect_sizes={},
                sample_size=100,
                seed=42,
                metadata={},
                raw_results=[]
            )

            assert 0.0 <= result.available_fraction <= 1.0, \
                f"available_fraction {result.available_fraction} not in [0, 1] for scores {scores}"


class TestEvalResultMetadata:
    """Test EvalResult metadata handling."""

    def test_metadata_preservation(self):
        """Test that metadata is preserved correctly."""
        metadata = {
            "suite_config": {"name": "test"},
            "run_data_shape": (100, 5),
            "sampled_data_shape": (50, 5),
            "evaluation_count": 3,
            "failed_evaluations": [],
            "normalized_columns": {"global_step": "step"}
        }

        result = EvalResult(
            suite_name="test",
            scores={},
            confidence_intervals={},
            effect_sizes={},
            sample_size=50,
            seed=42,
            metadata=metadata,
            raw_results=[]
        )

        assert result.metadata == metadata

    def test_metadata_contains_expected_keys(self):
        """Test that metadata contains expected keys from evaluation run."""
        result = EvalResult(
            suite_name="test",
            scores={"metric1": 0.8},
            confidence_intervals={"metric1": (0.7, 0.9)},
            effect_sizes={"metric1": 0.5},
            sample_size=100,
            seed=42,
            metadata={
                "suite_config": {"name": "test"},
                "run_data_shape": (100, 5),
                "sampled_data_shape": (100, 5),
                "evaluation_count": 1,
                "failed_evaluations": [],
                "normalized_columns": {}
            },
            raw_results=[{"evaluation": "metric1", "result": {"score": 0.8}}]
        )

        expected_keys = [
            "suite_config",
            "run_data_shape",
            "sampled_data_shape",
            "evaluation_count",
            "failed_evaluations",
            "normalized_columns"
        ]

        for key in expected_keys:
            assert key in result.metadata, f"Missing key: {key}"


class TestEvalResultWarnings:
    """Test EvalResult warnings handling."""

    def test_warnings_are_list(self):
        """Test that warnings is always a list."""
        result = EvalResult(
            suite_name="test",
            scores={},
            confidence_intervals={},
            effect_sizes={},
            sample_size=0,
            seed=42,
            metadata={},
            raw_results=[]
        )

        assert isinstance(result.warnings, list)

    def test_warnings_can_be_modified(self):
        """Test that warnings list can be modified."""
        result = EvalResult(
            suite_name="test",
            scores={},
            confidence_intervals={},
            effect_sizes={},
            sample_size=0,
            seed=42,
            metadata={},
            raw_results=[]
        )

        # Should start empty
        assert result.warnings == []

        # Should be able to append
        result.warnings.append("new warning")
        assert len(result.warnings) == 1
        assert "new warning" in result.warnings

    def test_warnings_from_evaluation_run(self):
        """Test warnings from actual evaluation run."""
        warnings = [
            "events column not provided, event-based diagnostics will be skipped",
            "metric1 failed to compute",
            "DataFrame contains only NaN values"
        ]

        result = EvalResult(
            suite_name="test",
            scores={"metric2": 0.8},
            confidence_intervals={},
            effect_sizes={},
            sample_size=100,
            seed=42,
            metadata={},
            raw_results=[],
            warnings=warnings
        )

        assert len(result.warnings) == 3
        for warning in warnings:
            assert warning in result.warnings


class TestEvalResultConsistency:
    """Test EvalResult internal consistency."""

    def test_scores_and_available_fraction_consistency(self):
        """Test that available_fraction matches actual valid scores."""
        scores = {"a": 0.8, "b": np.nan, "c": 0.6, "d": None}

        result = EvalResult(
            suite_name="test",
            scores=scores,
            confidence_intervals={},
            effect_sizes={},
            sample_size=100,
            seed=42,
            metadata={},
            raw_results=[]
        )

        # Count valid scores manually
        valid_count = 0
        total_count = len(scores)

        for score in scores.values():
            if score is not None and not (isinstance(score, float) and np.isnan(score)):
                valid_count += 1

        expected_fraction = valid_count / total_count
        assert abs(result.available_fraction - expected_fraction) < 1e-10

    def test_overall_score_and_available_fraction_consistency(self):
        """Test that overall_score is None when available_fraction is 0."""
        result = EvalResult(
            suite_name="test",
            scores={"a": np.nan, "b": None},
            confidence_intervals={},
            effect_sizes={},
            sample_size=100,
            seed=42,
            metadata={},
            raw_results=[]
        )

        assert result.available_fraction == 0.0
        assert result.overall_score is None

    def test_overall_score_and_available_fraction_consistency_positive(self):
        """Test that overall_score is not None when available_fraction > 0."""
        result = EvalResult(
            suite_name="test",
            scores={"a": 0.8, "b": np.nan, "c": 0.6},
            confidence_intervals={},
            effect_sizes={},
            sample_size=100,
            seed=42,
            metadata={},
            raw_results=[]
        )

        assert result.available_fraction > 0.0
        assert result.overall_score is not None
        assert result.overall_score == 0.7  # (0.8 + 0.6) / 2
