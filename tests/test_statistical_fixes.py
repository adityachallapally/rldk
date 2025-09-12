"""Tests for statistical method fixes to ensure accuracy and correctness."""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any

# Import the modules we're testing
from src.rldk.evals.metrics.utils import calculate_confidence_intervals, calculate_effect_sizes
from src.rldk.evals.metrics import calculate_confidence_intervals as metrics_ci, calculate_effect_sizes as metrics_es
from src.rldk.cards.reward import _calculate_calibration_score
from src.rldk.evals.integrity import _calculate_severity_score, _normalize_integrity_score
from src.rldk.evals.probes import _reward_based_fallback, _stability_based_fallback


class TestConfidenceIntervals:
    """Test confidence interval calculations."""
    
    def test_with_actual_data(self):
        """Test confidence intervals with actual sample data."""
        # Generate known data
        np.random.seed(42)
        sample_data = {
            'accuracy': np.random.binomial(100, 0.7, 50) / 100,  # Binary-like data
            'reward': np.random.normal(0.5, 0.2, 50)  # Continuous data
        }
        
        scores = {
            'accuracy': np.mean(sample_data['accuracy']),
            'reward': np.mean(sample_data['reward'])
        }
        
        # Test with sample data
        ci_with_data = calculate_confidence_intervals(
            scores, sample_size=50, sample_data=sample_data
        )
        
        # Test without sample data (fallback)
        ci_without_data = calculate_confidence_intervals(
            scores, sample_size=50
        )
        
        # With actual data should be more accurate
        for metric in scores:
            ci_with = ci_with_data[metric]
            ci_without = ci_without_data[metric]
            
            # Both should be valid intervals
            assert ci_with[0] <= scores[metric] <= ci_with[1]
            assert ci_without[0] <= scores[metric] <= ci_without[1]
            
            # With actual data should generally be tighter (more accurate)
            width_with = ci_with[1] - ci_with[0]
            width_without = ci_without[1] - ci_without[0]
            assert width_with <= width_without * 1.5  # Allow some tolerance
    
    def test_edge_cases(self):
        """Test edge cases for confidence intervals."""
        # Single sample
        ci_single = calculate_confidence_intervals({'test': 0.5}, sample_size=1)
        assert ci_single['test'] == (0.5, 0.5)
        
        # Zero sample size
        ci_zero = calculate_confidence_intervals({'test': 0.5}, sample_size=0)
        assert ci_zero['test'] == (0.5, 0.5)
        
        # NaN score
        ci_nan = calculate_confidence_intervals({'test': np.nan}, sample_size=10)
        assert np.isnan(ci_nan['test'][0]) and np.isnan(ci_nan['test'][1])


class TestEffectSizes:
    """Test effect size calculations."""
    
    def test_with_actual_data(self):
        """Test effect sizes with actual sample data."""
        np.random.seed(42)
        
        # Generate two groups with known difference
        group1_data = np.random.normal(0.5, 0.1, 100)
        group2_data = np.random.normal(0.7, 0.1, 100)  # Higher mean
        
        sample_data = {'metric': group1_data}
        baseline_sample_data = {'metric': group2_data}
        
        scores = {'metric': np.mean(group1_data)}
        baseline_scores = {'metric': np.mean(group2_data)}
        
        # Test with sample data
        es_with_data = calculate_effect_sizes(
            scores, baseline_scores, sample_data, baseline_sample_data
        )
        
        # Test without sample data (fallback)
        es_without_data = calculate_effect_sizes(scores, baseline_scores)
        
        # Both should give reasonable effect sizes
        assert not np.isnan(es_with_data['metric'])
        assert not np.isnan(es_without_data['metric'])
        
        # With actual data should be more accurate
        assert abs(es_with_data['metric']) >= abs(es_without_data['metric']) * 0.5
    
    def test_binary_metrics(self):
        """Test effect sizes for binary metrics."""
        scores = {'accuracy': 0.8}
        baseline_scores = {'accuracy': 0.6}
        
        es = calculate_effect_sizes(scores, baseline_scores)
        
        # Should give reasonable effect size for binary data
        assert not np.isnan(es['accuracy'])
        assert abs(es['accuracy']) > 0  # Should detect the difference


class TestCalibrationScore:
    """Test calibration score calculation."""
    
    def test_data_driven_calibration(self):
        """Test that calibration score adapts to data."""
        # Create mock events with different reward patterns
        events_data = [
            {'reward_mean': 0.5, 'reward_std': 0.1},  # Well-calibrated
            {'reward_mean': 0.6, 'reward_std': 0.12},  # Well-calibrated
            {'reward_mean': 0.4, 'reward_std': 0.08},  # Well-calibrated
        ]
        
        df = pd.DataFrame(events_data)
        
        # Mock the events_to_dataframe function
        class MockEvent:
            def __init__(self, metrics):
                self.metrics = metrics
        
        events = [MockEvent(row) for _, row in df.iterrows()]
        
        # Test calibration score
        score = _calculate_calibration_score(events)
        
        # Should give reasonable score
        assert 0 <= score <= 1
        assert not np.isnan(score)


class TestIntegrityScoring:
    """Test integrity scoring system."""
    
    def test_severity_scoring(self):
        """Test severity score calculation."""
        # Test escalating penalties
        score1 = _calculate_severity_score(0.05, [(0.1, 0.2), (0.2, 0.3)])
        score2 = _calculate_severity_score(0.15, [(0.1, 0.2), (0.2, 0.3)])
        score3 = _calculate_severity_score(0.25, [(0.1, 0.2), (0.2, 0.3)])
        
        # Should escalate properly
        assert score1 == 0.0  # Below first threshold
        assert score2 == 0.2  # Above first threshold
        assert score3 == 0.5  # Above both thresholds
    
    def test_normalization(self):
        """Test score normalization."""
        # Test normal cases
        norm1 = _normalize_integrity_score(0.3, 1.0)
        assert norm1 == 0.3
        
        # Test edge cases
        norm2 = _normalize_integrity_score(0.0, 0.0)  # Division by zero
        assert norm2 == 0.5  # Should return neutral
        
        # Test capping
        norm3 = _normalize_integrity_score(1.5, 1.0)
        assert norm3 == 1.0  # Should be capped


class TestFallbackLogic:
    """Test fallback logic improvements."""
    
    def test_reward_based_fallback(self):
        """Test reward-based fallback logic."""
        # Test with different reward ranges
        df1 = pd.DataFrame({'reward_mean': [0.1, 0.2, 0.3]})  # Range 0.2
        df2 = pd.DataFrame({'reward_mean': [0.8, 0.9, 1.0]})  # Range 0.2
        df3 = pd.DataFrame({'reward_mean': [0.0, 0.5, 1.0]})  # Range 1.0
        
        score1 = _reward_based_fallback(df1)
        score2 = _reward_based_fallback(df2)
        score3 = _reward_based_fallback(df3)
        
        # All should be valid scores
        for score in [score1, score2, score3]:
            assert 0 <= score <= 1
            assert not np.isnan(score)
        
        # Higher rewards should generally give higher scores
        assert score2 > score1
    
    def test_stability_based_fallback(self):
        """Test stability-based fallback logic."""
        # Test with different stability levels
        df_stable = pd.DataFrame({'reward_mean': [0.5, 0.51, 0.49, 0.5]})  # Stable
        df_unstable = pd.DataFrame({'reward_mean': [0.1, 0.9, 0.2, 0.8]})  # Unstable
        
        score_stable = _stability_based_fallback(df_stable)
        score_unstable = _stability_based_fallback(df_unstable)
        
        # Stable should score higher
        assert score_stable > score_unstable
        
        # Both should be valid
        for score in [score_stable, score_unstable]:
            assert 0 <= score <= 1
            assert not np.isnan(score)


class TestBackwardCompatibility:
    """Test that changes maintain backward compatibility."""
    
    def test_confidence_intervals_compatibility(self):
        """Test that old API still works."""
        scores = {'test': 0.5}
        
        # Old API (without sample_data)
        ci_old = calculate_confidence_intervals(scores, sample_size=10)
        
        # Should work without errors
        assert 'test' in ci_old
        assert len(ci_old['test']) == 2
        assert ci_old['test'][0] <= ci_old['test'][1]
    
    def test_effect_sizes_compatibility(self):
        """Test that old API still works."""
        scores = {'test': 0.5}
        baseline_scores = {'test': 0.4}
        
        # Old API (without sample_data)
        es_old = calculate_effect_sizes(scores, baseline_scores)
        
        # Should work without errors
        assert 'test' in es_old
        assert not np.isnan(es_old['test'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])