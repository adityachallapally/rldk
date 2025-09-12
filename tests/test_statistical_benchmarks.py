"""Statistical benchmarks against known libraries to validate our implementations."""

import pytest
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.weightstats import ttest_ind
import warnings

# Import our implementations
from src.rldk.evals.metrics.utils import calculate_confidence_intervals, calculate_effect_sizes
from src.rldk.evals.metrics import calculate_confidence_intervals as metrics_ci


class TestConfidenceIntervalBenchmarks:
    """Benchmark confidence intervals against scipy and statsmodels."""
    
    def test_binomial_confidence_intervals(self):
        """Test binomial confidence intervals against statsmodels."""
        np.random.seed(42)
        
        # Generate binomial data
        n_trials = 1000
        true_prop = 0.7
        successes = np.random.binomial(n_trials, true_prop, 100)
        proportions = successes / n_trials
        
        for prop in proportions:
            # Our implementation
            our_ci = calculate_confidence_intervals(
                {'accuracy': prop}, 
                sample_size=n_trials,
                sample_data={'accuracy': np.array([prop] * n_trials)}
            )
            
            # Statsmodels implementation (Wilson method)
            sm_ci = proportion_confint(
                int(prop * n_trials), 
                n_trials, 
                alpha=0.05, 
                method='wilson'
            )
            
            # Compare results (allow some tolerance)
            our_lower, our_upper = our_ci['accuracy']
            sm_lower, sm_upper = sm_ci
            
            # Our CI should be within reasonable bounds of statsmodels
            assert abs(our_lower - sm_lower) < 0.05, f"Lower bound differs: {our_lower} vs {sm_lower}"
            assert abs(our_upper - sm_upper) < 0.05, f"Upper bound differs: {our_upper} vs {sm_upper}"
    
    def test_normal_confidence_intervals(self):
        """Test normal confidence intervals against scipy."""
        np.random.seed(42)
        
        # Generate normal data
        true_mean = 0.5
        true_std = 0.2
        sample_size = 100
        data = np.random.normal(true_mean, true_std, sample_size)
        
        # Our implementation
        our_ci = calculate_confidence_intervals(
            {'reward': np.mean(data)}, 
            sample_size=sample_size,
            sample_data={'reward': data}
        )
        
        # Scipy implementation
        scipy_ci = stats.t.interval(
            0.95, 
            df=sample_size-1, 
            loc=np.mean(data), 
            scale=stats.sem(data)
        )
        
        # Compare results
        our_lower, our_upper = our_ci['reward']
        scipy_lower, scipy_upper = scipy_ci
        
        # Should be very close (within numerical precision)
        assert abs(our_lower - scipy_lower) < 1e-10, f"Lower bound differs: {our_lower} vs {scipy_lower}"
        assert abs(our_upper - scipy_upper) < 1e-10, f"Upper bound differs: {our_upper} vs {scipy_upper}"
    
    def test_effect_size_benchmarks(self):
        """Test effect sizes against scipy t-test."""
        np.random.seed(42)
        
        # Generate two groups with known difference
        group1 = np.random.normal(0.5, 0.1, 100)
        group2 = np.random.normal(0.7, 0.1, 100)  # Higher mean
        
        # Our implementation
        our_es = calculate_effect_sizes(
            {'metric': np.mean(group1)}, 
            {'metric': np.mean(group2)},
            sample_data={'metric': group1},
            baseline_sample_data={'metric': group2}
        )
        
        # Scipy implementation (Cohen's d)
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                             (len(group2) - 1) * np.var(group2, ddof=1)) / 
                            (len(group1) + len(group2) - 2))
        scipy_es = (np.mean(group1) - np.mean(group2)) / pooled_std
        
        # Compare results
        our_value = our_es['metric']
        
        # Should be very close (within numerical precision)
        assert abs(our_value - scipy_es) < 1e-10, f"Effect size differs: {our_value} vs {scipy_es}"
    
    def test_fallback_methods_accuracy(self):
        """Test that fallback methods are reasonably accurate."""
        np.random.seed(42)
        
        # Test with known data
        true_mean = 0.5
        true_std = 0.2
        sample_size = 50
        data = np.random.normal(true_mean, true_std, sample_size)
        
        # Method 1: With actual data (gold standard)
        ci_with_data = calculate_confidence_intervals(
            {'metric': np.mean(data)}, 
            sample_size=sample_size,
            sample_data={'metric': data}
        )
        
        # Method 2: Fallback method
        ci_fallback = calculate_confidence_intervals(
            {'metric': np.mean(data)}, 
            sample_size=sample_size
        )
        
        # Method 3: Legacy method
        ci_legacy = calculate_confidence_intervals(
            {'metric': np.mean(data)}, 
            sample_size=sample_size,
            use_legacy_method=True
        )
        
        # All methods should give reasonable results
        for method_name, ci in [("with_data", ci_with_data), ("fallback", ci_fallback), ("legacy", ci_legacy)]:
            lower, upper = ci['metric']
            assert lower <= np.mean(data) <= upper, f"{method_name} CI doesn't contain mean"
            assert upper - lower > 0, f"{method_name} CI has zero width"
            assert upper - lower < 1.0, f"{method_name} CI is too wide"


class TestStatisticalAssumptions:
    """Test that our statistical assumptions are valid."""
    
    def test_binomial_assumption_validity(self):
        """Test that binomial assumption is only applied to appropriate metrics."""
        # Metrics that should use binomial assumption
        binomial_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Metrics that should NOT use binomial assumption
        continuous_metrics = ['reward', 'loss', 'gradient_norm', 'learning_rate']
        
        for metric in binomial_metrics:
            score = 0.8  # In [0, 1] range
            ci = calculate_confidence_intervals({metric: score}, sample_size=100)
            
            # Should use binomial standard deviation
            # This is validated by checking if the CI width is reasonable for binomial
            lower, upper = ci[metric]
            width = upper - lower
            
            # Binomial CI should be narrower than conservative estimate
            assert width < 0.2, f"Binomial CI too wide for {metric}: {width}"
        
        for metric in continuous_metrics:
            score = 2.5  # Outside [0, 1] range
            ci = calculate_confidence_intervals({metric: score}, sample_size=100)
            
            # Should use conservative estimate
            lower, upper = ci[metric]
            width = upper - lower
            
            # Conservative CI should be reasonable but not too narrow
            assert 0.1 < width < 0.5, f"Conservative CI width inappropriate for {metric}: {width}"
    
    def test_sample_size_handling(self):
        """Test that different sample sizes are handled appropriately."""
        score = 0.5
        
        # Small sample (should be conservative)
        ci_small = calculate_confidence_intervals({'metric': score}, sample_size=5)
        width_small = ci_small['metric'][1] - ci_small['metric'][0]
        
        # Large sample (should be more precise)
        ci_large = calculate_confidence_intervals({'metric': score}, sample_size=1000)
        width_large = ci_large['metric'][1] - ci_large['metric'][0]
        
        # Large sample should give narrower CI
        assert width_large < width_small, f"Large sample CI not narrower: {width_large} vs {width_small}"
        
        # But both should be reasonable
        assert width_small < 0.8, f"Small sample CI too wide: {width_small}"
        assert width_large > 0.01, f"Large sample CI too narrow: {width_large}"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_variance(self):
        """Test handling of zero variance data."""
        # All values are the same
        data = np.array([0.5] * 100)
        
        ci = calculate_confidence_intervals(
            {'metric': 0.5}, 
            sample_size=100,
            sample_data={'metric': data}
        )
        
        # Should handle gracefully
        lower, upper = ci['metric']
        assert not np.isnan(lower) and not np.isnan(upper)
        assert lower <= 0.5 <= upper
    
    def test_extreme_values(self):
        """Test handling of extreme values."""
        # Very small values
        ci_small = calculate_confidence_intervals({'metric': 0.001}, sample_size=100)
        assert not np.isnan(ci_small['metric'][0]) and not np.isnan(ci_small['metric'][1])
        
        # Very large values
        ci_large = calculate_confidence_intervals({'metric': 1000.0}, sample_size=100)
        assert not np.isnan(ci_large['metric'][0]) and not np.isnan(ci_large['metric'][1])
        
        # Zero
        ci_zero = calculate_confidence_intervals({'metric': 0.0}, sample_size=100)
        assert not np.isnan(ci_zero['metric'][0]) and not np.isnan(ci_zero['metric'][1])
    
    def test_single_sample(self):
        """Test handling of single sample."""
        ci = calculate_confidence_intervals({'metric': 0.5}, sample_size=1)
        
        # Should return point estimate
        lower, upper = ci['metric']
        assert lower == upper == 0.5
    
    def test_zero_sample_size(self):
        """Test handling of zero sample size."""
        ci = calculate_confidence_intervals({'metric': 0.5}, sample_size=0)
        
        # Should return point estimate
        lower, upper = ci['metric']
        assert lower == upper == 0.5


class TestPerformanceBenchmarks:
    """Benchmark performance of our implementations."""
    
    def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        import time
        
        # Large dataset
        n_samples = 10000
        data = np.random.normal(0.5, 0.2, n_samples)
        
        start_time = time.time()
        ci = calculate_confidence_intervals(
            {'metric': np.mean(data)}, 
            sample_size=n_samples,
            sample_data={'metric': data}
        )
        end_time = time.time()
        
        # Should complete quickly (under 1 second)
        assert end_time - start_time < 1.0, f"Too slow: {end_time - start_time:.2f}s"
        
        # Should give reasonable result
        lower, upper = ci['metric']
        assert lower < np.mean(data) < upper
    
    def test_memory_usage(self):
        """Test memory usage doesn't grow excessively."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process many datasets
        for i in range(100):
            data = np.random.normal(0.5, 0.2, 1000)
            ci = calculate_confidence_intervals(
                {'metric': np.mean(data)}, 
                sample_size=1000,
                sample_data={'metric': data}
            )
        
        final_memory = process.memory_info().rss
        memory_growth = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory growth should be reasonable (under 100MB)
        assert memory_growth < 100, f"Excessive memory growth: {memory_growth:.1f}MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])