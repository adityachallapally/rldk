"""Comprehensive unit tests for KL divergence numeric stability and edge cases."""

import os
import sys
import warnings
from unittest.mock import patch

import numpy as np
import pytest

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Import only the specific modules we need, avoiding the full package import
from rldk.evals.metrics import (
    calculate_kl_divergence,
    calculate_kl_divergence_between_runs,
)
from rldk.forensics.kl_schedule_tracker import (
    KLScheduleMetrics,
    KLScheduleTracker,
    _safe_coefficient_value,
    _safe_kl_value,
)


class TestKLDivergenceNumericStability:
    """Test KL divergence calculation for comprehensive numeric stability."""

    def test_zeros_handling(self):
        """Test handling of zero distributions."""
        # Both distributions are zero
        p = np.array([0.0, 0.0, 0.0])
        q = np.array([0.0, 0.0, 0.0])
        kl_div = calculate_kl_divergence(p, q)
        assert kl_div == 0.0

        # P is zero, Q is not
        p = np.array([0.0, 0.0, 0.0])
        q = np.array([0.3, 0.3, 0.4])
        kl_div = calculate_kl_divergence(p, q)
        assert kl_div == 0.0

        # Q is zero, P is not
        p = np.array([0.3, 0.3, 0.4])
        q = np.array([0.0, 0.0, 0.0])
        kl_div = calculate_kl_divergence(p, q)
        assert np.isinf(kl_div)

    def test_identical_distributions(self):
        """Test KL divergence between identical distributions."""
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.5, 0.3, 0.2])

        kl_div = calculate_kl_divergence(p, q)
        assert abs(kl_div) < 1e-6  # Should be very close to 0

    def test_negative_inputs_rejected(self):
        """Test that negative inputs are properly rejected."""
        p = np.array([0.5, -0.1, 0.3])
        q = np.array([0.4, 0.4, 0.2])

        with pytest.raises(ValueError, match="Probability distributions must be non-negative"):
            calculate_kl_divergence(p, q)

        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.4, -0.1, 0.2])

        with pytest.raises(ValueError, match="Probability distributions must be non-negative"):
            calculate_kl_divergence(p, q)

    def test_tiny_values_handling(self):
        """Test handling of very small probability values."""
        # Very small probabilities
        p = np.array([1e-12, 0.5, 0.5])
        q = np.array([1e-10, 0.4, 0.6])

        kl_div = calculate_kl_divergence(p, q)
        assert isinstance(kl_div, float)
        assert kl_div >= 0.0
        assert not np.isnan(kl_div)
        assert not np.isinf(kl_div)

        # Extremely small probabilities
        p = np.array([1e-20, 0.5, 0.5])
        q = np.array([1e-18, 0.4, 0.6])

        kl_div = calculate_kl_divergence(p, q)
        assert isinstance(kl_div, float)
        assert kl_div >= 0.0
        assert not np.isnan(kl_div)
        assert not np.isinf(kl_div)

    def test_huge_values_handling(self):
        """Test handling of very large probability values."""
        # Large probabilities
        p = np.array([1000.0, 2000.0, 3000.0])
        q = np.array([1200.0, 1800.0, 3000.0])

        kl_div = calculate_kl_divergence(p, q)
        assert isinstance(kl_div, float)
        assert kl_div >= 0.0
        assert not np.isnan(kl_div)
        assert not np.isinf(kl_div)

        # Extremely large probabilities
        p = np.array([1e6, 2e6, 3e6])
        q = np.array([1.2e6, 1.8e6, 3e6])

        kl_div = calculate_kl_divergence(p, q)
        assert isinstance(kl_div, float)
        assert kl_div >= 0.0
        assert not np.isnan(kl_div)
        assert not np.isinf(kl_div)

    def test_nan_inputs_rejected(self):
        """Test that NaN inputs are properly rejected."""
        p = np.array([0.5, np.nan, 0.3])
        q = np.array([0.4, 0.4, 0.2])

        with pytest.raises(ValueError, match="Input distributions contain NaN values"):
            calculate_kl_divergence(p, q)

        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.4, np.nan, 0.2])

        with pytest.raises(ValueError, match="Input distributions contain NaN values"):
            calculate_kl_divergence(p, q)

    def test_inf_inputs_rejected(self):
        """Test that infinite inputs are properly rejected."""
        p = np.array([0.5, np.inf, 0.3])
        q = np.array([0.4, 0.4, 0.2])

        with pytest.raises(ValueError, match="Input distributions contain infinite values"):
            calculate_kl_divergence(p, q)

        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.4, np.inf, 0.2])

        with pytest.raises(ValueError, match="Input distributions contain infinite values"):
            calculate_kl_divergence(p, q)

    def test_fallback_epsilon_path(self):
        """Test the fallback epsilon path when primary calculation fails."""
        # Create distributions that would cause numerical issues
        p = np.array([1e-20, 0.5, 0.5])
        q = np.array([1e-25, 0.4, 0.6])

        # This should trigger the fallback epsilon path
        kl_div = calculate_kl_divergence(p, q)
        assert isinstance(kl_div, float)
        assert kl_div >= 0.0
        assert not np.isnan(kl_div)
        assert not np.isinf(kl_div)

    def test_capping_extreme_values(self):
        """Test that extremely large KL divergence values are capped."""
        # Create distributions that would lead to very large KL divergence
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([1e-20, 0.5, 0.5])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            kl_div = calculate_kl_divergence(p, q)

            # Should be capped at 1e6
            assert kl_div <= 1e6
            assert len(w) >= 1
            assert any("extremely large" in str(warning.message).lower() for warning in w)

    def test_length_mismatch_rejected(self):
        """Test that mismatched array lengths are properly rejected."""
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.4, 0.4])

        with pytest.raises(ValueError, match="Distributions must have the same length"):
            calculate_kl_divergence(p, q)

    def test_normalization_stability(self):
        """Test that normalization is numerically stable."""
        # Test with very small sums
        p = np.array([1e-10, 1e-10, 1e-10])
        q = np.array([2e-10, 2e-10, 2e-10])

        kl_div = calculate_kl_divergence(p, q)
        assert isinstance(kl_div, float)
        assert kl_div >= 0.0
        assert not np.isnan(kl_div)
        assert not np.isinf(kl_div)

        # Test with very large sums
        p = np.array([1e6, 1e6, 1e6])
        q = np.array([2e6, 2e6, 2e6])

        kl_div = calculate_kl_divergence(p, q)
        assert isinstance(kl_div, float)
        assert kl_div >= 0.0
        assert not np.isnan(kl_div)
        assert not np.isinf(kl_div)

    def test_repeated_calculations_consistency(self):
        """Test that repeated calculations produce consistent results."""
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.4, 0.4, 0.2])

        results = []
        for _ in range(100):
            kl_div = calculate_kl_divergence(p, q)
            results.append(kl_div)

        # All results should be identical (within numerical precision)
        assert all(abs(r - results[0]) < 1e-12 for r in results)

    def test_edge_case_single_element(self):
        """Test KL divergence with single-element distributions."""
        p = np.array([1.0])
        q = np.array([1.0])

        kl_div = calculate_kl_divergence(p, q)
        assert kl_div == 0.0

        p = np.array([1.0])
        q = np.array([0.5])

        kl_div = calculate_kl_divergence(p, q)
        assert isinstance(kl_div, float)
        assert kl_div >= 0.0
        assert not np.isnan(kl_div)
        assert not np.isinf(kl_div)


class TestKLDivergenceBetweenRuns:
    """Test KL divergence between runs functionality."""

    def test_basic_between_runs_calculation(self):
        """Test basic KL divergence between runs calculation."""
        run1_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        run2_data = np.array([1.1, 2.1, 2.9, 4.1, 4.9])

        result = calculate_kl_divergence_between_runs(run1_data, run2_data)

        assert isinstance(result, dict)
        assert "kl_divergence" in result
        assert isinstance(result["kl_divergence"], float)
        assert not np.isnan(result["kl_divergence"])
        assert not np.isinf(result["kl_divergence"])

    def test_empty_data_handling(self):
        """Test handling of empty data in between runs calculation."""
        run1_data = np.array([])
        run2_data = np.array([1.0, 2.0, 3.0])

        result = calculate_kl_divergence_between_runs(run1_data, run2_data)

        assert isinstance(result, dict)
        assert np.isnan(result["kl_divergence"])
        assert "error" in result
        assert "Insufficient data" in result["error"]

    def test_nan_data_handling(self):
        """Test handling of NaN data in between runs calculation."""
        run1_data = np.array([1.0, np.nan, 3.0, 4.0])
        run2_data = np.array([1.1, 2.1, np.nan, 4.1])

        result = calculate_kl_divergence_between_runs(run1_data, run2_data)

        assert isinstance(result, dict)
        assert "kl_divergence" in result
        # Should handle NaN values by filtering them out
        assert not np.isnan(result["kl_divergence"])

    def test_dataframe_input_handling(self):
        """Test handling of DataFrame inputs."""
        import pandas as pd

        df1 = pd.DataFrame({"reward_mean": [1.0, 2.0, 3.0, 4.0]})
        df2 = pd.DataFrame({"reward_mean": [1.1, 2.1, 2.9, 4.1]})

        result = calculate_kl_divergence_between_runs(df1, df2, metric="reward_mean")

        assert isinstance(result, dict)
        assert "kl_divergence" in result
        assert not np.isnan(result["kl_divergence"])

    def test_different_bins_parameter(self):
        """Test KL divergence calculation with different bin counts."""
        run1_data = np.random.normal(0, 1, 1000)
        run2_data = np.random.normal(0.1, 1, 1000)

        result_10_bins = calculate_kl_divergence_between_runs(run1_data, run2_data, bins=10)
        result_50_bins = calculate_kl_divergence_between_runs(run1_data, run2_data, bins=50)

        assert isinstance(result_10_bins["kl_divergence"], float)
        assert isinstance(result_50_bins["kl_divergence"], float)
        assert not np.isnan(result_10_bins["kl_divergence"])
        assert not np.isnan(result_50_bins["kl_divergence"])


class TestSafeValueProcessing:
    """Test safe value processing functions for comprehensive edge case handling."""

    def test_safe_kl_value_comprehensive(self):
        """Test safe KL value processing with comprehensive edge cases."""
        # Valid inputs
        assert _safe_kl_value(0.5) == 0.5
        assert _safe_kl_value(1.0) == 1.0
        assert _safe_kl_value(0) == 0.0
        assert _safe_kl_value("0.5") == 0.5
        assert _safe_kl_value(1000) == 1000

        # None values
        assert _safe_kl_value(None) == 0.0

        # NaN values
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _safe_kl_value(np.nan)
            assert result == 0.0
            assert len(w) == 1
            assert "NaN KL value detected" in str(w[0].message)

        # Infinite values
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _safe_kl_value(np.inf)
            assert result == 1e6
            assert len(w) == 1
            assert "Positive infinity KL value detected" in str(w[0].message)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _safe_kl_value(-np.inf)
            assert result == 0.0
            assert len(w) == 1
            assert "Negative infinity KL value detected" in str(w[0].message)

        # Negative values
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _safe_kl_value(-0.1)
            assert result == 0.0
            assert len(w) == 1
            assert "Negative KL value" in str(w[0].message)

        # Extremely large values
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _safe_kl_value(2e6)
            assert result == 1e6
            assert len(w) == 1
            assert "Extremely large KL value" in str(w[0].message)

        # Invalid string values
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _safe_kl_value("invalid")
            assert result == 0.0
            assert len(w) == 1
            assert "Could not convert string" in str(w[0].message)

        # Non-numeric types
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _safe_kl_value([1, 2, 3])
            assert result == 0.0
            assert len(w) == 1
            assert "Non-numeric KL value type" in str(w[0].message)

    def test_safe_coefficient_value_comprehensive(self):
        """Test safe coefficient value processing with comprehensive edge cases."""
        # Valid inputs
        assert _safe_coefficient_value(1.0) == 1.0
        assert _safe_coefficient_value(0.5) == 0.5
        assert _safe_coefficient_value("1.0") == 1.0
        assert _safe_coefficient_value(1000) == 1000

        # None values
        assert _safe_coefficient_value(None) == 1.0

        # Zero and negative values
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _safe_coefficient_value(0.0)
            assert result == 1.0
            assert len(w) == 1
            assert "Non-positive coefficient value" in str(w[0].message)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _safe_coefficient_value(-0.1)
            assert result == 1.0
            assert len(w) == 1
            assert "Non-positive coefficient value" in str(w[0].message)

        # Very small values
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _safe_coefficient_value(1e-10)
            assert result == 1e-8
            assert len(w) == 1
            assert "Very small coefficient value" in str(w[0].message)

        # Extremely large values
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _safe_coefficient_value(2e6)
            assert result == 1e6
            assert len(w) == 1
            assert "Extremely large coefficient value" in str(w[0].message)

        # NaN values
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _safe_coefficient_value(np.nan)
            assert result == 1.0
            assert len(w) == 1
            assert "NaN coefficient value detected" in str(w[0].message)

        # Infinite values
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _safe_coefficient_value(np.inf)
            assert result == 1e6
            assert len(w) == 1
            assert "Positive infinity coefficient value detected" in str(w[0].message)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _safe_coefficient_value(-np.inf)
            assert result == 1.0
            assert len(w) == 1
            assert "Negative infinity coefficient value detected" in str(w[0].message)


class TestKLScheduleTrackerRobustness:
    """Test KL schedule tracker robustness with comprehensive edge cases."""

    def test_tracker_initialization(self):
        """Test tracker initialization."""
        tracker = KLScheduleTracker(kl_target=0.1, kl_target_tolerance=0.05)
        assert tracker.kl_target == 0.1
        assert tracker.kl_target_tolerance == 0.05
        assert len(tracker.kl_history) == 0
        assert len(tracker.kl_coef_history) == 0

    def test_tracker_update_comprehensive_edge_cases(self):
        """Test tracker update with comprehensive edge cases."""
        tracker = KLScheduleTracker()

        # Test with valid values
        metrics = tracker.update(step=1, kl_value=0.05, kl_coef=1.0)
        assert isinstance(metrics, KLScheduleMetrics)
        assert metrics.current_kl == 0.05
        assert metrics.current_kl_coef == 1.0

        # Test with NaN values
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            metrics = tracker.update(step=2, kl_value=np.nan, kl_coef=np.nan)
            assert metrics.current_kl == 0.0
            assert metrics.current_kl_coef == 1.0
            assert len(w) >= 2  # Should have warnings for both NaN values

        # Test with infinite values
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            metrics = tracker.update(step=3, kl_value=np.inf, kl_coef=np.inf)
            assert metrics.current_kl == 1e6
            assert metrics.current_kl_coef == 1e6
            assert len(w) >= 2  # Should have warnings for both inf values

        # Test with negative values
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            metrics = tracker.update(step=4, kl_value=-0.1, kl_coef=-0.1)
            assert metrics.current_kl == 0.0
            assert metrics.current_kl_coef == 1.0
            assert len(w) >= 2  # Should have warnings for both negative values

        # Test with string values
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            metrics = tracker.update(step=5, kl_value="0.2", kl_coef="1.5")
            assert metrics.current_kl == 0.2
            assert metrics.current_kl_coef == 1.5

        # Test with None values
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            metrics = tracker.update(step=6, kl_value=None, kl_coef=None)
            assert metrics.current_kl == 0.0
            assert metrics.current_kl_coef == 1.0

    def test_tracker_analysis_robustness(self):
        """Test tracker analysis functions with various data patterns."""
        tracker = KLScheduleTracker()

        # Add some normal data
        for i in range(20):
            tracker.update(step=i, kl_value=0.1 + 0.01 * np.sin(i), kl_coef=1.0)

        # Add some extreme values
        tracker.update(step=20, kl_value=np.inf, kl_coef=np.nan)
        tracker.update(step=21, kl_value=-1.0, kl_coef=0.0)

        # Should not crash and should produce valid metrics
        metrics = tracker.get_summary()
        assert isinstance(metrics, dict)
        assert "current_kl" in metrics
        assert "current_kl_coef" in metrics
        assert "kl_health_score" in metrics
        assert "schedule_health_score" in metrics

    def test_tracker_anomaly_detection(self):
        """Test tracker anomaly detection."""
        tracker = KLScheduleTracker()

        # Add normal data
        for i in range(10):
            tracker.update(step=i, kl_value=0.1, kl_coef=1.0)

        # Add anomalous data
        tracker.update(step=10, kl_value=0.5, kl_coef=1.0)  # High KL

        anomalies = tracker.get_anomalies()
        assert isinstance(anomalies, list)
        # Should detect high KL anomaly
        assert len(anomalies) > 0
        assert any(anomaly["type"] == "kl_volatility_anomaly" for anomaly in anomalies)

    def test_tracker_consistency(self):
        """Test that tracker produces consistent results."""
        KLScheduleTracker()

        # Add same data multiple times
        for _ in range(3):
            tracker_copy = KLScheduleTracker()
            for i in range(10):
                tracker_copy.update(step=i, kl_value=0.1 + 0.01 * i, kl_coef=1.0 + 0.1 * i)

            summary = tracker_copy.get_summary()
            assert isinstance(summary, dict)
            assert "current_kl" in summary
            assert "current_kl_coef" in summary


class TestNumericalStabilityRegression:
    """Test for numerical stability regressions."""

    def test_repeated_calculations_consistency(self):
        """Test that repeated calculations produce consistent results."""
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.4, 0.4, 0.2])

        results = []
        for _ in range(100):
            kl_div = calculate_kl_divergence(p, q)
            results.append(kl_div)

        # All results should be identical (within numerical precision)
        assert all(abs(r - results[0]) < 1e-12 for r in results)

    def test_tracker_consistency(self):
        """Test that tracker produces consistent results."""
        KLScheduleTracker()

        # Add same data multiple times
        for _ in range(3):
            tracker_copy = KLScheduleTracker()
            for i in range(10):
                tracker_copy.update(step=i, kl_value=0.1 + 0.01 * i, kl_coef=1.0 + 0.1 * i)

            summary = tracker_copy.get_summary()
            assert isinstance(summary, dict)
            assert "current_kl" in summary
            assert "current_kl_coef" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
