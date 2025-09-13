"""Tests for robust division by zero handling in RLDK."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

from src.rldk.utils.math_utils import (
    try_divide, safe_percentage, safe_rate, 
    nan_aware_mean, nan_aware_std, nan_aware_median
)
from src.rldk.evals.metrics.throughput import (
    calculate_tokens_per_second, evaluate_throughput
)
from src.rldk.evals.suites import (
    evaluate_consistency, evaluate_robustness, evaluate_efficiency
)


class TestMathUtils:
    """Test the math utilities for robust division."""
    
    def test_try_divide_positive_denominator(self):
        """Test division with positive denominator."""
        result, used = try_divide(10, 2)
        assert result == 5.0
        assert used is True
    
    def test_try_divide_zero_denominator_skip(self):
        """Test division by zero with skip behavior."""
        result, used = try_divide(10, 0, on_zero="skip")
        assert np.isnan(result)
        assert used is False
    
    def test_try_divide_zero_denominator_zero(self):
        """Test division by zero with zero behavior."""
        result, used = try_divide(10, 0, on_zero="zero")
        assert result == 0.0
        assert used is True
    
    def test_try_divide_zero_denominator_nan(self):
        """Test division by zero with NaN behavior."""
        result, used = try_divide(10, 0, on_zero="nan")
        assert np.isnan(result)
        assert used is True
    
    def test_try_divide_negative_denominator(self):
        """Test division with negative denominator (treated as zero)."""
        result, used = try_divide(10, -1, on_zero="skip")
        assert np.isnan(result)
        assert used is False
    
    def test_safe_percentage_positive_denominator(self):
        """Test safe percentage calculation with positive denominator."""
        percentage, used, reason = safe_percentage(25, 100)
        assert percentage == 25.0
        assert used is True
        assert reason == "calculated_successfully"
    
    def test_safe_percentage_zero_denominator_skip(self):
        """Test safe percentage with zero denominator and skip behavior."""
        percentage, used, reason = safe_percentage(25, 0, on_zero="skip")
        assert np.isnan(percentage)
        assert used is False
        assert reason == "zero_denominator_skipped"
    
    def test_safe_percentage_zero_denominator_zero(self):
        """Test safe percentage with zero denominator and zero behavior."""
        percentage, used, reason = safe_percentage(25, 0, on_zero="zero")
        assert percentage == 0.0
        assert used is True
        assert reason == "zero_denominator_as_zero"
    
    def test_safe_rate_positive_denominator(self):
        """Test safe rate calculation with positive denominator."""
        rate, used, reason = safe_rate(100, 10)
        assert rate == 10.0
        assert used is True
        assert reason == "calculated_successfully"
    
    def test_safe_rate_zero_denominator_skip(self):
        """Test safe rate with zero denominator and skip behavior."""
        rate, used, reason = safe_rate(100, 0, on_zero="skip")
        assert np.isnan(rate)
        assert used is False
        assert reason == "zero_denominator_skipped"
    
    def test_nan_aware_mean_empty_list(self):
        """Test nan_aware_mean with empty list."""
        result = nan_aware_mean([])
        assert np.isnan(result)
    
    def test_nan_aware_mean_with_nan(self):
        """Test nan_aware_mean with NaN values."""
        values = [1.0, 2.0, float('nan'), 4.0, 5.0]
        result = nan_aware_mean(values)
        assert result == 3.0  # (1+2+4+5)/4
    
    def test_nan_aware_mean_all_nan(self):
        """Test nan_aware_mean with all NaN values."""
        values = [float('nan'), float('nan'), float('nan')]
        result = nan_aware_mean(values)
        assert np.isnan(result)
    
    def test_nan_aware_std_with_nan(self):
        """Test nan_aware_std with NaN values."""
        values = [1.0, 2.0, float('nan'), 4.0, 5.0]
        result = nan_aware_std(values)
        expected = np.std([1.0, 2.0, 4.0, 5.0], ddof=1)
        assert abs(result - expected) < 1e-10
    
    def test_nan_aware_median_with_nan(self):
        """Test nan_aware_median with NaN values."""
        values = [1.0, 2.0, float('nan'), 4.0, 5.0]
        result = nan_aware_median(values)
        assert result == 3.0  # median of [1, 2, 4, 5]


class TestThroughputMetrics:
    """Test throughput metrics with robust division."""
    
    def test_calculate_tokens_per_second_empty_events(self):
        """Test calculate_tokens_per_second with empty events."""
        mean_tps, std_tps, total_tokens, counters = calculate_tokens_per_second([])
        assert mean_tps == 0.0
        assert std_tps == 0.0
        assert total_tokens == 0.0
        assert counters["samples_seen"] == 0
        assert counters["samples_used"] == 0
    
    def test_calculate_tokens_per_second_with_zero_time_intervals(self):
        """Test calculate_tokens_per_second with zero time intervals."""
        events = [
            {"timestamp": 0, "token_count": 10},
            {"timestamp": 0, "token_count": 20},  # Same timestamp = zero interval
            {"timestamp": 1, "token_count": 30},
        ]
        
        mean_tps, std_tps, total_tokens, counters = calculate_tokens_per_second(events)
        
        # Should skip the zero interval
        assert counters["non_positive_time_skipped"] == 1
        assert counters["samples_seen"] == 2
        assert counters["samples_used"] == 1  # Only one valid interval
        assert total_tokens == 50  # 20 + 30
    
    def test_calculate_tokens_per_second_with_processing_time_zero(self):
        """Test calculate_tokens_per_second with zero processing time."""
        events = [
            {"timestamp": 0, "batch_size": 10, "processing_time": 1.0},
            {"timestamp": 1, "batch_size": 20, "processing_time": 0.0},  # Zero processing time
            {"timestamp": 2, "batch_size": 30, "processing_time": 2.0},
        ]
        
        mean_tps, std_tps, total_tokens, counters = calculate_tokens_per_second(events)
        
        # Should skip the zero processing time
        assert counters["zero_denominator_skipped"] >= 1
        assert counters["samples_used"] >= 1
    
    def test_evaluate_throughput_with_counters(self):
        """Test evaluate_throughput returns counters."""
        # Create test data with event logs
        data = pd.DataFrame({
            "events": [
                json.dumps([
                    {"timestamp": 0, "token_count": 10},
                    {"timestamp": 1, "token_count": 20},
                    {"timestamp": 2, "token_count": 30},
                ])
            ]
        })
        
        result = evaluate_throughput(data)
        
        assert "counters" in result
        assert "samples_seen" in result["counters"]
        assert "samples_used" in result["counters"]
        assert "zero_denominator_skipped" in result["counters"]
        assert "non_positive_time_skipped" in result["counters"]
        assert "other_skip_reasons" in result["counters"]


class TestSuiteEvaluationFunctions:
    """Test suite evaluation functions with robust division."""
    
    def test_evaluate_consistency_with_zero_mean(self):
        """Test evaluate_consistency handles zero mean values."""
        data = pd.DataFrame({
            "reward_mean": [0.0, 0.0, 0.0, 0.0, 0.0],  # All zeros
            "step": [1, 2, 3, 4, 5]
        })
        
        result = evaluate_consistency(data)
        
        # Should not crash and return a valid score
        assert "score" in result
        assert isinstance(result["score"], float)
        assert 0.0 <= result["score"] <= 1.0
    
    def test_evaluate_robustness_with_zero_mean(self):
        """Test evaluate_robustness handles zero mean values."""
        data = pd.DataFrame({
            "reward_mean": [0.0, 0.0, 0.0, 0.0, 0.0],  # All zeros
            "step": [1, 2, 3, 4, 5]
        })
        
        result = evaluate_robustness(data)
        
        # Should not crash and return a valid score
        assert "score" in result
        assert isinstance(result["score"], float)
        assert 0.0 <= result["score"] <= 1.0
    
    def test_evaluate_efficiency_with_zero_time(self):
        """Test evaluate_efficiency handles zero time values."""
        data = pd.DataFrame({
            "training_time": [0.0, 0.0, 0.0, 0.0, 0.0],  # All zeros
            "step": [1, 2, 3, 4, 5]
        })
        
        result = evaluate_efficiency(data)
        
        # Should not crash and return a valid score
        assert "score" in result
        assert isinstance(result["score"], float)
        assert 0.0 <= result["score"] <= 1.0


class TestPropertyTests:
    """Property tests for division by zero handling."""
    
    def test_random_division_by_zero_injection(self):
        """Test that randomly injected zeros don't cause exceptions."""
        np.random.seed(42)
        
        # Generate random data
        n_samples = 100
        data = pd.DataFrame({
            "reward_mean": np.random.normal(0.5, 0.1, n_samples),
            "step": range(n_samples),
            "training_time": np.random.uniform(0.1, 1.0, n_samples),
        })
        
        # Randomly inject zeros in denominators
        zero_indices = np.random.choice(n_samples, size=10, replace=False)
        data.loc[zero_indices, "training_time"] = 0.0
        
        # Test that functions don't crash
        consistency_result = evaluate_consistency(data)
        robustness_result = evaluate_robustness(data)
        efficiency_result = evaluate_efficiency(data)
        
        # All should return valid scores
        assert 0.0 <= consistency_result["score"] <= 1.0
        assert 0.0 <= robustness_result["score"] <= 1.0
        assert 0.0 <= efficiency_result["score"] <= 1.0
    
    def test_nan_propagation_consistency(self):
        """Test that NaN values are handled consistently across functions."""
        # Create data with some NaN values
        data = pd.DataFrame({
            "reward_mean": [1.0, 2.0, float('nan'), 4.0, 5.0],
            "step": [1, 2, 3, 4, 5],
            "training_time": [0.1, 0.2, 0.0, 0.4, 0.5],  # One zero
        })
        
        # All functions should handle NaN and zero denominators gracefully
        consistency_result = evaluate_consistency(data)
        robustness_result = evaluate_robustness(data)
        efficiency_result = evaluate_efficiency(data)
        
        # Results should be valid numbers (not NaN or inf)
        assert not np.isnan(consistency_result["score"])
        assert not np.isnan(robustness_result["score"])
        assert not np.isnan(efficiency_result["score"])
        assert not np.isinf(consistency_result["score"])
        assert not np.isinf(robustness_result["score"])
        assert not np.isinf(efficiency_result["score"])


class TestAdvancedMonitoring:
    """Test advanced monitoring with robust division."""
    
    def test_custom_callback_division_counters(self):
        """Test that CustomRLDKCallback tracks division counters."""
        from examples.trl_integration.advanced_monitoring import CustomRLDKCallback
        
        callback = CustomRLDKCallback(output_dir="./test_output", run_id="test")
        
        # Check initial counters
        assert callback.division_counters["samples_seen"] == 0
        assert callback.division_counters["samples_used"] == 0
        assert callback.division_counters["zero_denominator_skipped"] == 0
    
    def test_throughput_metrics_with_counters(self):
        """Test that throughput metrics include counters."""
        from examples.trl_integration.advanced_monitoring import CustomRLDKCallback
        
        callback = CustomRLDKCallback(output_dir="./test_output", run_id="test")
        
        # Get throughput metrics
        throughput_metrics = callback.get_throughput_metrics()
        
        assert "tokens_per_second" in throughput_metrics
        assert "window_size_used" in throughput_metrics
        assert "zero_time_samples_skipped" in throughput_metrics
        assert "division_counters" in throughput_metrics


if __name__ == "__main__":
    pytest.main([__file__])