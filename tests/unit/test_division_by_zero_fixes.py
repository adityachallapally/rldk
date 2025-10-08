"""Test division by zero fixes across the codebase."""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rldk.utils.error_handling import (
    safe_divide,
    safe_percentage,
    safe_rate_calculation,
    safe_ratio,
)


class TestSafeDivisionFunctions:
    """Test the safe division utility functions."""

    def test_safe_divide_normal_case(self):
        """Test normal division case."""
        result = safe_divide(10, 2)
        assert result == 5.0

    def test_safe_divide_by_zero_default_fallback(self):
        """Test division by zero with default fallback."""
        result = safe_divide(10, 0)
        assert result == 0.0

    def test_safe_divide_by_zero_custom_fallback(self):
        """Test division by zero with custom fallback."""
        result = safe_divide(10, 0, fallback=1.0)
        assert result == 1.0

    def test_safe_rate_calculation_normal_case(self):
        """Test normal rate calculation."""
        result = safe_rate_calculation(100, 10)
        assert result == 10.0

    def test_safe_rate_calculation_by_zero(self):
        """Test rate calculation with zero time interval."""
        result = safe_rate_calculation(100, 0)
        assert result == 0.0

    def test_safe_percentage_normal_case(self):
        """Test normal percentage calculation."""
        result = safe_percentage(25, 100)
        assert result == 25.0

    def test_safe_percentage_by_zero(self):
        """Test percentage calculation with zero denominator."""
        result = safe_percentage(25, 0)
        assert result == 0.0

    def test_safe_ratio_normal_case(self):
        """Test normal ratio calculation."""
        result = safe_ratio(3, 4)
        assert result == 0.75

    def test_safe_ratio_by_zero(self):
        """Test ratio calculation with zero denominator."""
        result = safe_ratio(3, 0)
        assert result == 0.0


class TestThroughputMetricsDivisionByZero:
    """Test division by zero fixes in throughput metrics."""

    def test_token_rate_calculation_with_zero_time_interval(self):
        """Test that token rate calculation handles zero time interval."""
        from rldk.evals.metrics.throughput import calculate_token_throughput

        # Create events with zero time interval
        events = [
            {"timestamp": 0, "token_count": 100},
            {"timestamp": 0, "token_count": 200},  # Same timestamp = zero interval
        ]

        # This should not raise a division by zero error
        mean_rate, std_rate, total_tokens = calculate_token_throughput(events)

        # Should return valid results
        assert isinstance(mean_rate, float)
        assert isinstance(std_rate, float)
        assert isinstance(total_tokens, float)

    def test_batch_processing_with_zero_processing_time(self):
        """Test batch processing with zero processing time."""
        from rldk.evals.metrics.throughput import calculate_batch_throughput

        # Create events with zero processing time
        events = [
            {"timestamp": 0, "batch_size": 100, "processing_time": 0},
            {"timestamp": 1, "batch_size": 200, "processing_time": 0},
        ]

        # This should not raise a division by zero error
        result = calculate_batch_throughput(events)

        # Should return valid results
        assert isinstance(result, dict)


class TestSuitesDivisionByZero:
    """Test division by zero fixes in evaluation suites."""

    def test_batch_speed_calculation_with_zero_batch_times(self):
        """Test batch speed calculation with zero batch times."""
        from rldk.evals.suites import evaluate_speed

        # Create data with zero batch times
        data = pd.DataFrame({
            "batch_time": [0, 0, 0, 1, 2],
            "batch_size": [100, 200, 300, 400, 500],
            "step": [1, 2, 3, 4, 5],
            "training_time": [0, 1, 2, 3, 4]
        })

        # This should not raise a division by zero error
        result = evaluate_speed(data)

        # Should return valid results
        assert isinstance(result, dict)
        assert "speed_metrics" in result

    def test_training_speed_calculation_with_zero_time(self):
        """Test training speed calculation with zero total time."""
        from rldk.evals.suites import evaluate_efficiency

        # Create data with zero total time
        data = pd.DataFrame({
            "step": [1, 2, 3, 4, 5],
            "training_time": [0, 0, 0, 0, 0],  # All same time = zero total time
            "memory_usage": [100, 200, 300, 400, 500]
        })

        # This should not raise a division by zero error
        result = evaluate_efficiency(data)

        # Should return valid results
        assert isinstance(result, dict)
        assert "efficiency_metrics" in result


class TestAdvancedMonitoringDivisionByZero:
    """Test division by zero fixes in advanced monitoring."""

    def test_tokens_per_second_calculation_with_zero_time(self):
        """Test tokens per second calculation with zero total time."""
        from examples.trl_integration.advanced_monitoring import (
            AdvancedMonitoringCallback,
        )

        # Create mock metrics with zero step time
        class MockMetric:
            def __init__(self, tokens_in, tokens_out, step_time):
                self.tokens_in = tokens_in
                self.tokens_out = tokens_out
                self.step_time = step_time

        callback = AdvancedMonitoringCallback()

        # Add metrics with zero step time
        callback.metrics_history = [
            MockMetric(100, 200, 0),
            MockMetric(150, 250, 0),
            MockMetric(200, 300, 0),
        ]

        # This should not raise a division by zero error
        callback._calculate_tokens_per_second()

        # Should set tokens_per_second to 0.0
        assert callback.current_metrics.tokens_per_second == 0.0


class TestProgressDivisionByZero:
    """Test division by zero fixes in progress utilities."""

    def test_download_progress_with_zero_elapsed_time(self):
        """Test download progress with zero elapsed time."""
        from rldk.utils.progress import create_download_progress

        # Create progress callback
        callback = create_download_progress(1000, "Test Download")

        # Call with zero elapsed time (immediately after start)
        # This should not raise a division by zero error
        callback(100)  # 100 bytes downloaded

        # Should handle gracefully without errors
        assert True  # If we get here without exception, test passes


class TestComprehensiveDivisionByZeroProtection:
    """Test comprehensive protection against division by zero."""

    def test_all_safe_division_functions(self):
        """Test all safe division functions with edge cases."""
        test_cases = [
            # (numerator, denominator, expected_result)
            (10, 2, 5.0),
            (10, 0, 0.0),
            (0, 5, 0.0),
            (0, 0, 0.0),
            (-10, 2, -5.0),
            (10, -2, 0.0),  # Now returns fallback for negative denominators
            (-10, -2, 0.0),  # Now returns fallback for negative denominators
        ]

        for num, denom, expected in test_cases:
            result = safe_divide(num, denom)
            assert result == expected, f"safe_divide({num}, {denom}) = {result}, expected {expected}"

    def test_rate_calculation_edge_cases(self):
        """Test rate calculation with various edge cases."""
        test_cases = [
            # (count, time_interval, expected_result)
            (100, 10, 10.0),
            (100, 0, 0.0),
            (0, 10, 0.0),
            (0, 0, 0.0),
            (-100, 10, -10.0),
            (100, -10, 0.0),  # Now returns fallback for negative denominators
        ]

        for count, time_interval, expected in test_cases:
            result = safe_rate_calculation(count, time_interval)
            assert result == expected, f"safe_rate_calculation({count}, {time_interval}) = {result}, expected {expected}"

    def test_percentage_calculation_edge_cases(self):
        """Test percentage calculation with various edge cases."""
        test_cases = [
            # (numerator, denominator, expected_result)
            (25, 100, 25.0),
            (25, 0, 0.0),
            (0, 100, 0.0),
            (0, 0, 0.0),
            (50, 200, 25.0),
            (25, -100, 0.0),  # Now returns fallback for negative denominators
        ]

        for num, denom, expected in test_cases:
            result = safe_percentage(num, denom)
            assert result == expected, f"safe_percentage({num}, {denom}) = {result}, expected {expected}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
