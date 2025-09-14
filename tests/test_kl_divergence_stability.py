"""Comprehensive tests for KL divergence calculation stability and edge cases."""

import os
import sys
import warnings
from unittest.mock import patch

import numpy as np
import pytest

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rldk.evals.metrics import calculate_kl_divergence
from rldk.forensics.kl_schedule_tracker import (
    KLScheduleMetrics,
    KLScheduleTracker,
    _safe_coefficient_value,
    _safe_kl_value,
)


class TestKLDivergenceStability:
    """Test KL divergence calculation for numerical stability."""

    def test_basic_kl_divergence(self):
        """Test basic KL divergence calculation."""
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.4, 0.4, 0.2])

        kl_div = calculate_kl_divergence(p, q)
        assert isinstance(kl_div, float)
        assert kl_div >= 0.0
        assert not np.isnan(kl_div)
        assert not np.isinf(kl_div)

    def test_identical_distributions(self):
        """Test KL divergence between identical distributions."""
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.5, 0.3, 0.2])

        kl_div = calculate_kl_divergence(p, q)
        assert abs(kl_div) < 1e-6  # Should be very close to 0

    def test_zero_probabilities(self):
        """Test KL divergence with zero probabilities."""
        p = np.array([0.0, 0.5, 0.5])
        q = np.array([0.1, 0.4, 0.5])

        kl_div = calculate_kl_divergence(p, q)
        assert isinstance(kl_div, float)
        assert kl_div >= 0.0
        assert not np.isnan(kl_div)
        assert not np.isinf(kl_div)

    def test_very_small_probabilities(self):
        """Test KL divergence with very small probabilities."""
        p = np.array([1e-10, 0.5, 0.5])
        q = np.array([1e-12, 0.4, 0.6])

        kl_div = calculate_kl_divergence(p, q)
        assert isinstance(kl_div, float)
        assert kl_div >= 0.0
        assert not np.isnan(kl_div)
        assert not np.isinf(kl_div)

    def test_large_probabilities(self):
        """Test KL divergence with large probability values."""
        p = np.array([1000.0, 2000.0, 3000.0])
        q = np.array([1200.0, 1800.0, 3000.0])

        kl_div = calculate_kl_divergence(p, q)
        assert isinstance(kl_div, float)
        assert kl_div >= 0.0
        assert not np.isnan(kl_div)
        assert not np.isinf(kl_div)

    def test_nan_input_handling(self):
        """Test handling of NaN inputs."""
        p = np.array([0.5, np.nan, 0.3])
        q = np.array([0.4, 0.4, 0.2])

        with pytest.raises(ValueError, match="Input distributions contain NaN values"):
            calculate_kl_divergence(p, q)

    def test_inf_input_handling(self):
        """Test handling of infinite inputs."""
        p = np.array([0.5, np.inf, 0.3])
        q = np.array([0.4, 0.4, 0.2])

        with pytest.raises(ValueError, match="Input distributions contain infinite values"):
            calculate_kl_divergence(p, q)

    def test_negative_input_handling(self):
        """Test handling of negative inputs."""
        p = np.array([0.5, -0.1, 0.3])
        q = np.array([0.4, 0.4, 0.2])

        with pytest.raises(ValueError, match="Probability distributions must be non-negative"):
            calculate_kl_divergence(p, q)

    def test_zero_sum_distributions(self):
        """Test handling of zero-sum distributions."""
        p = np.array([0.0, 0.0, 0.0])
        q = np.array([0.0, 0.0, 0.0])

        kl_div = calculate_kl_divergence(p, q)
        assert kl_div == 0.0

    def test_mismatched_lengths(self):
        """Test handling of mismatched array lengths."""
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.4, 0.4])

        with pytest.raises(ValueError, match="Distributions must have the same length"):
            calculate_kl_divergence(p, q)

    def test_extreme_kl_values(self):
        """Test handling of extreme KL divergence values."""
        # Create distributions that would lead to very large KL divergence
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([1e-10, 0.5, 0.5])

        kl_div = calculate_kl_divergence(p, q)
        assert isinstance(kl_div, float)
        assert kl_div >= 0.0
        assert not np.isnan(kl_div)
        # Should be capped at 1e6
        assert kl_div <= 1e6


class TestSafeValueProcessing:
    """Test safe value processing functions."""

    def test_safe_kl_value_valid_inputs(self):
        """Test safe KL value processing with valid inputs."""
        assert _safe_kl_value(0.5) == 0.5
        assert _safe_kl_value(1.0) == 1.0
        assert _safe_kl_value(0) == 0.0
        assert _safe_kl_value("0.5") == 0.5
        assert _safe_kl_value(1000) == 1000

    def test_safe_kl_value_edge_cases(self):
        """Test safe KL value processing with edge cases."""
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

    def test_safe_coefficient_value_valid_inputs(self):
        """Test safe coefficient value processing with valid inputs."""
        assert _safe_coefficient_value(1.0) == 1.0
        assert _safe_coefficient_value(0.5) == 0.5
        assert _safe_coefficient_value("1.0") == 1.0
        assert _safe_coefficient_value(1000) == 1000

    def test_safe_coefficient_value_edge_cases(self):
        """Test safe coefficient value processing with edge cases."""
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


class TestKLScheduleTrackerRobustness:
    """Test KL schedule tracker robustness."""

    def test_tracker_initialization(self):
        """Test tracker initialization."""
        tracker = KLScheduleTracker(kl_target=0.1, kl_target_tolerance=0.05)
        assert tracker.kl_target == 0.1
        assert tracker.kl_target_tolerance == 0.05
        assert len(tracker.kl_history) == 0
        assert len(tracker.kl_coef_history) == 0

    def test_tracker_update_valid_values(self):
        """Test tracker update with valid values."""
        tracker = KLScheduleTracker()

        metrics = tracker.update(step=1, kl_value=0.05, kl_coef=1.0)
        assert isinstance(metrics, KLScheduleMetrics)
        assert metrics.current_kl == 0.05
        assert metrics.current_kl_coef == 1.0
        assert len(tracker.kl_history) == 1

    def test_tracker_update_edge_cases(self):
        """Test tracker update with edge cases."""
        tracker = KLScheduleTracker()

        # Test with NaN values
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            metrics = tracker.update(step=1, kl_value=np.nan, kl_coef=np.nan)
            assert metrics.current_kl == 0.0
            assert metrics.current_kl_coef == 1.0
            assert len(w) >= 2  # Should have warnings for both NaN values

        # Test with infinite values
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            metrics = tracker.update(step=2, kl_value=np.inf, kl_coef=np.inf)
            assert metrics.current_kl == 1e6
            assert metrics.current_kl_coef == 1e6
            assert len(w) >= 2  # Should have warnings for both inf values

        # Test with negative values
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            metrics = tracker.update(step=3, kl_value=-0.1, kl_coef=-0.1)
            assert metrics.current_kl == 0.0
            assert metrics.current_kl_coef == 1.0
            assert len(w) >= 2  # Should have warnings for both negative values

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


class TestNumericalStabilityRegression:
    """Test for numerical stability regressions."""

    def test_repeated_calculations(self):
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
