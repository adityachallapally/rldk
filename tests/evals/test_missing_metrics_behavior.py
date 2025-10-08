"""Tests for missing metrics behavior and overall score computation."""

import numpy as np
import pandas as pd
import pytest

from rldk.evals.runner import EvalResult
from rldk.evals.schema import safe_mean


class TestEvalResultOverallScore:
    """Test EvalResult overall_score property."""

    def test_overall_score_with_valid_metrics(self):
        """Test overall_score with valid metrics."""
        result = EvalResult(
            suite_name="test",
            scores={"metric1": 0.8, "metric2": 0.6, "metric3": 0.9},
            confidence_intervals={},
            effect_sizes={},
            sample_size=100,
            seed=42,
            metadata={},
            raw_results=[]
        )

        assert result.overall_score == 0.7666666666666666  # (0.8 + 0.6 + 0.9) / 3
        assert result.available_fraction == 1.0

    def test_overall_score_with_some_nan_metrics(self):
        """Test overall_score with some NaN metrics."""
        result = EvalResult(
            suite_name="test",
            scores={"metric1": 0.8, "metric2": np.nan, "metric3": 0.9},
            confidence_intervals={},
            effect_sizes={},
            sample_size=100,
            seed=42,
            metadata={},
            raw_results=[]
        )

        assert abs(result.overall_score - 0.85) < 1e-10  # (0.8 + 0.9) / 2, with floating point tolerance
        assert result.available_fraction == 2/3  # 2 out of 3 metrics available

    def test_overall_score_with_all_nan_metrics(self):
        """Test overall_score with all NaN metrics."""
        result = EvalResult(
            suite_name="test",
            scores={"metric1": np.nan, "metric2": np.nan, "metric3": np.nan},
            confidence_intervals={},
            effect_sizes={},
            sample_size=100,
            seed=42,
            metadata={},
            raw_results=[]
        )

        assert result.overall_score is None
        assert result.available_fraction == 0.0

    def test_overall_score_with_none_metrics(self):
        """Test overall_score with None metrics."""
        result = EvalResult(
            suite_name="test",
            scores={"metric1": None, "metric2": None, "metric3": None},
            confidence_intervals={},
            effect_sizes={},
            sample_size=100,
            seed=42,
            metadata={},
            raw_results=[]
        )

        assert result.overall_score is None
        assert result.available_fraction == 0.0

    def test_overall_score_with_mixed_nan_and_none(self):
        """Test overall_score with mixed NaN and None values."""
        result = EvalResult(
            suite_name="test",
            scores={"metric1": 0.8, "metric2": np.nan, "metric3": None},
            confidence_intervals={},
            effect_sizes={},
            sample_size=100,
            seed=42,
            metadata={},
            raw_results=[]
        )

        assert result.overall_score == 0.8  # Only metric1 is valid
        assert result.available_fraction == 1/3  # 1 out of 3 metrics available

    def test_overall_score_empty_scores(self):
        """Test overall_score with empty scores dictionary."""
        result = EvalResult(
            suite_name="test",
            scores={},
            confidence_intervals={},
            effect_sizes={},
            sample_size=100,
            seed=42,
            metadata={},
            raw_results=[]
        )

        assert result.overall_score is None
        assert result.available_fraction == 0.0

    def test_available_fraction_calculation(self):
        """Test available_fraction calculation in various scenarios."""
        # Test with all valid metrics
        result1 = EvalResult(
            suite_name="test",
            scores={"a": 0.5, "b": 0.6, "c": 0.7},
            confidence_intervals={},
            effect_sizes={},
            sample_size=100,
            seed=42,
            metadata={},
            raw_results=[]
        )
        assert result1.available_fraction == 1.0

        # Test with half valid metrics
        result2 = EvalResult(
            suite_name="test",
            scores={"a": 0.5, "b": np.nan, "c": 0.7, "d": None},
            confidence_intervals={},
            effect_sizes={},
            sample_size=100,
            seed=42,
            metadata={},
            raw_results=[]
        )
        assert result2.available_fraction == 0.5  # 2 out of 4 metrics available

        # Test with no valid metrics
        result3 = EvalResult(
            suite_name="test",
            scores={"a": np.nan, "b": None},
            confidence_intervals={},
            effect_sizes={},
            sample_size=100,
            seed=42,
            metadata={},
            raw_results=[]
        )
        assert result3.available_fraction == 0.0


class TestSafeMeanFunction:
    """Test safe_mean function behavior."""

    def test_safe_mean_valid_values(self):
        """Test safe_mean with valid numeric values."""
        values = [1.0, 2.0, 3.0, 4.0]
        result = safe_mean(values)
        assert result == 2.5

    def test_safe_mean_empty_list(self):
        """Test safe_mean with empty list."""
        result = safe_mean([])
        assert result is None

    def test_safe_mean_with_nan(self):
        """Test safe_mean filters out NaN values."""
        values = [1.0, np.nan, 3.0, np.nan, 5.0]
        result = safe_mean(values)
        assert result == 3.0  # (1.0 + 3.0 + 5.0) / 3

    def test_safe_mean_all_nan(self):
        """Test safe_mean with all NaN values."""
        values = [np.nan, np.nan, np.nan]
        result = safe_mean(values)
        assert result is None

    def test_safe_mean_with_none(self):
        """Test safe_mean filters out None values."""
        values = [1.0, None, 3.0, None, 5.0]
        result = safe_mean(values)
        assert result == 3.0  # (1.0 + 3.0 + 5.0) / 3

    def test_safe_mean_all_none(self):
        """Test safe_mean with all None values."""
        values = [None, None, None]
        result = safe_mean(values)
        assert result is None

    def test_safe_mean_mixed_invalid(self):
        """Test safe_mean with mixed NaN and None values."""
        values = [1.0, np.nan, 3.0, None, 5.0, np.nan]
        result = safe_mean(values)
        assert result == 3.0  # (1.0 + 3.0 + 5.0) / 3

    def test_safe_mean_single_value(self):
        """Test safe_mean with single valid value."""
        values = [42.0]
        result = safe_mean(values)
        assert result == 42.0

    def test_safe_mean_single_nan(self):
        """Test safe_mean with single NaN value."""
        values = [np.nan]
        result = safe_mean(values)
        assert result is None


class TestMissingMetricsBehavior:
    """Test behavior when metrics cannot be computed."""

    def test_evaluation_result_with_no_metrics(self):
        """Test EvalResult when no metrics can be computed."""
        result = EvalResult(
            suite_name="test",
            scores={"metric1": None, "metric2": None},
            confidence_intervals={},
            effect_sizes={},
            sample_size=0,
            seed=42,
            metadata={"failed_evaluations": ["metric1", "metric2"]},
            raw_results=[],
            warnings=["No valid metrics computed - check data quality"]
        )

        assert result.overall_score is None
        assert result.available_fraction == 0.0
        assert len(result.warnings) == 1
        assert "No valid metrics computed" in result.warnings[0]

    def test_evaluation_result_with_partial_metrics(self):
        """Test EvalResult when only some metrics can be computed."""
        result = EvalResult(
            suite_name="test",
            scores={"metric1": 0.8, "metric2": None, "metric3": 0.6},
            confidence_intervals={},
            effect_sizes={},
            sample_size=100,
            seed=42,
            metadata={"failed_evaluations": ["metric2"]},
            raw_results=[],
            warnings=["metric2 failed to compute"]
        )

        assert result.overall_score == 0.7  # (0.8 + 0.6) / 2
        assert result.available_fraction == 2/3  # 2 out of 3 metrics available
        assert len(result.warnings) == 1

    def test_evaluation_result_with_warnings(self):
        """Test EvalResult warnings handling."""
        warnings = [
            "events column not provided, event-based diagnostics will be skipped",
            "metric1 failed to compute"
        ]

        result = EvalResult(
            suite_name="test",
            scores={"metric2": 0.8, "metric3": 0.6},
            confidence_intervals={},
            effect_sizes={},
            sample_size=100,
            seed=42,
            metadata={},
            raw_results=[],
            warnings=warnings
        )

        assert result.overall_score == 0.7  # (0.8 + 0.6) / 2
        assert len(result.warnings) == 2
        assert warnings[0] in result.warnings
        assert warnings[1] in result.warnings


class TestDefaultScoringRemoval:
    """Test that default 0.5 scoring has been removed."""

    def test_no_silent_defaults(self):
        """Test that evaluations don't return silent default scores."""
        # This test ensures that when metrics can't be computed,
        # they return None instead of 0.5

        result = EvalResult(
            suite_name="test",
            scores={"metric1": None, "metric2": None},
            confidence_intervals={},
            effect_sizes={},
            sample_size=100,
            seed=42,
            metadata={},
            raw_results=[]
        )

        # Should be None, not 0.5
        assert result.overall_score is None
        assert result.available_fraction == 0.0

        # Individual scores should be None, not 0.5
        assert result.scores["metric1"] is None
        assert result.scores["metric2"] is None

    def test_available_fraction_precision(self):
        """Test that available_fraction is calculated precisely."""
        # Test various fractions
        test_cases = [
            ({"a": 1.0, "b": 2.0, "c": 3.0}, 1.0),  # All available
            ({"a": 1.0, "b": None, "c": 3.0}, 2/3),  # 2/3 available
            ({"a": 1.0, "b": np.nan, "c": 3.0}, 2/3),  # 2/3 available
            ({"a": None, "b": None, "c": 3.0}, 1/3),  # 1/3 available
            ({"a": None, "b": None, "c": None}, 0.0),  # None available
        ]

        for scores, expected_fraction in test_cases:
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

            assert abs(result.available_fraction - expected_fraction) < 1e-10, \
                f"Expected {expected_fraction}, got {result.available_fraction} for scores {scores}"
