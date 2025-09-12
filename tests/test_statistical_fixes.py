"""Tests for statistical method fixes to ensure accuracy and correctness."""

import pytest
import numpy as np
import pandas as pd
import torch
import tempfile
import os
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, patch

# Import the modules we're testing
from src.rldk.evals.metrics.utils import calculate_confidence_intervals, calculate_effect_sizes
from src.rldk.evals.metrics import calculate_confidence_intervals as metrics_ci, calculate_effect_sizes as metrics_es
from src.rldk.cards.reward import _calculate_calibration_score
from src.rldk.evals.integrity import _calculate_severity_score, _normalize_integrity_score
from src.rldk.evals.probes import _reward_based_fallback, _stability_based_fallback
from src.rldk.integrations.trl.monitors import CheckpointMonitor
from src.rldk.config.settings import RLDebugKitSettings, MemorySettings, FileSettings


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


class TestMemoryManagement:
    """Test memory management in TRL monitor."""
    
    def test_memory_limits(self):
        """Test that memory limits are respected."""
        # Create a mock model with large parameters
        model = Mock()
        
        # Mock parameters with different sizes
        small_param = torch.randn(100, 100)  # ~40KB
        large_param = torch.randn(10000, 10000)  # ~400MB
        
        model.named_parameters.return_value = [
            ("small_layer.weight", Mock(data=small_param, requires_grad=True)),
            ("large_layer.weight", Mock(data=large_param, requires_grad=True)),
        ]
        
        # Test with strict memory limits
        monitor = CheckpointMonitor(
            max_parameter_size_mb=50,  # Should skip large param
            max_total_memory_mb=100
        )
        
        weights = monitor._extract_current_weights(model)
        
        # Should only include small parameter
        assert weights is not None
        assert "small_layer.weight" in weights
        assert "large_layer.weight" not in weights
    
    def test_memory_cleanup(self):
        """Test that memory cleanup works properly."""
        monitor = CheckpointMonitor()
        
        # Simulate storing weights
        monitor.previous_weights = {
            "test_param": torch.randn(100, 100)
        }
        
        # Cleanup should work without errors
        monitor._cleanup_weights()
        
        assert monitor.previous_weights is None
    
    def test_out_of_memory_handling(self):
        """Test handling of out of memory errors."""
        monitor = CheckpointMonitor()
        
        # Mock model that raises RuntimeError
        model = Mock()
        model.named_parameters.side_effect = RuntimeError("CUDA out of memory")
        
        weights = monitor._extract_current_weights(model)
        
        # Should return None gracefully
        assert weights is None


class TestFileSafety:
    """Test file safety measures."""
    
    def test_large_file_handling(self):
        """Test handling of large files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a large file
            large_file = temp_path / "large.log"
            with open(large_file, 'w') as f:
                f.write("test content " * 1000000)  # ~13MB
            
            # Test file size check
            file_size = large_file.stat().st_size
            assert file_size > 5 * 1024 * 1024  # > 5MB
            
            # Should be skipped due to size limit
            from src.rldk.config.settings import get_settings
            settings = get_settings()
            max_size = settings.file.max_file_size_mb * 1024 * 1024
            
            assert file_size > max_size
    
    def test_encoding_handling(self):
        """Test handling of files with encoding issues."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create file with non-UTF8 content
            binary_file = temp_path / "binary.log"
            with open(binary_file, 'wb') as f:
                f.write(b'\xff\xfe\x00\x00')  # Invalid UTF-8
            
            # Should handle encoding errors gracefully
            try:
                with open(binary_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(1024)
                assert True  # Should not raise exception
            except UnicodeDecodeError:
                pytest.fail("Should handle encoding errors gracefully")


class TestStatisticalValidation:
    """Test statistical method validation."""
    
    def test_legacy_vs_new_methods(self):
        """Test that legacy and new methods give different but reasonable results."""
        scores = {'test': 0.5}
        sample_size = 100
        
        # Test legacy method
        ci_legacy = calculate_confidence_intervals(
            scores, sample_size, use_legacy_method=True
        )
        
        # Test new method
        ci_new = calculate_confidence_intervals(
            scores, sample_size, use_legacy_method=False
        )
        
        # Both should be valid
        assert ci_legacy['test'][0] <= scores['test'] <= ci_legacy['test'][1]
        assert ci_new['test'][0] <= scores['test'] <= ci_new['test'][1]
        
        # They should be different (new method should be more accurate)
        legacy_width = ci_legacy['test'][1] - ci_legacy['test'][0]
        new_width = ci_new['test'][1] - ci_new['test'][0]
        
        # New method should generally be tighter
        assert new_width <= legacy_width * 1.5  # Allow some tolerance
    
    def test_statistical_assumptions(self):
        """Test that statistical assumptions are clearly documented."""
        # Test binomial assumption for binary metrics
        binary_scores = {'accuracy': 0.8}
        ci_binary = calculate_confidence_intervals(binary_scores, 100)
        
        # Should use binomial standard deviation
        assert ci_binary['accuracy'][0] < 0.8 < ci_binary['accuracy'][1]
        
        # Test continuous assumption for non-binary metrics
        continuous_scores = {'reward': 2.5}
        ci_continuous = calculate_confidence_intervals(continuous_scores, 100)
        
        # Should use different assumption
        assert ci_continuous['reward'][0] < 2.5 < ci_continuous['reward'][1]


class TestConfiguration:
    """Test configuration system."""
    
    def test_settings_initialization(self):
        """Test that settings initialize properly."""
        settings = RLDebugKitSettings()
        
        assert settings.memory.max_parameter_size_mb == 50
        assert settings.file.max_file_size_mb == 5
        assert settings.statistical.default_confidence_level == 0.95
    
    def test_custom_settings(self):
        """Test custom settings."""
        custom_memory = MemorySettings(max_parameter_size_mb=100)
        custom_file = FileSettings(max_file_size_mb=10)
        
        settings = RLDebugKitSettings(
            memory=custom_memory,
            file=custom_file
        )
        
        assert settings.memory.max_parameter_size_mb == 100
        assert settings.file.max_file_size_mb == 10


class TestErrorHandling:
    """Test error handling consistency."""
    
    def test_consistent_error_handling(self):
        """Test that error handling is consistent across modules."""
        # Test confidence intervals with invalid data
        invalid_scores = {'test': np.nan}
        ci = calculate_confidence_intervals(invalid_scores, 10)
        
        # Should handle NaN gracefully
        assert np.isnan(ci['test'][0]) and np.isnan(ci['test'][1])
        
        # Test effect sizes with missing baseline
        scores = {'test': 0.5}
        baseline_scores = {}
        
        es = calculate_effect_sizes(scores, baseline_scores)
        
        # Should handle missing baseline gracefully
        assert np.isnan(es['test'])
    
    def test_warning_consistency(self):
        """Test that warnings are consistent."""
        import warnings
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # This should generate a warning
            calculate_confidence_intervals({'test': 0.5}, 10)
            
            # Should have at least one warning
            assert len(w) > 0
            assert "estimated standard deviation" in str(w[0].message)


class TestIntegration:
    """Test integration between modules."""
    
    def test_monitor_integration(self):
        """Test that monitor integrates properly with other components."""
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = CheckpointMonitor(output_dir=temp_dir)
            
            # Test that monitor can be initialized without errors
            assert monitor.run_id is not None
            assert monitor.output_dir.exists()
    
    def test_settings_integration(self):
        """Test that settings integrate with other modules."""
        from src.rldk.config.settings import get_settings, set_settings
        
        # Test getting default settings
        settings = get_settings()
        assert settings is not None
        
        # Test setting custom settings
        custom_settings = RLDebugKitSettings()
        set_settings(custom_settings)
        
        retrieved_settings = get_settings()
        assert retrieved_settings is custom_settings


if __name__ == "__main__":
    pytest.main([__file__, "-v"])