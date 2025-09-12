"""Integration tests for RL Debug Kit bug fixes."""

import pytest
import numpy as np
import pandas as pd
import torch
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch
import warnings

# Import modules for integration testing
from src.rldk.evals.metrics.utils import calculate_confidence_intervals, calculate_effect_sizes
from src.rldk.evals.metrics import calculate_confidence_intervals as metrics_ci
from src.rldk.cards.determinism import generate_determinism_card
from src.rldk.integrations.trl.monitors import CheckpointMonitor
from src.rldk.config.settings import ExtendedConfigSchema, load_settings_from_env
from src.rldk.evals.probes import evaluate_alignment, evaluate_helpfulness
from src.rldk.evals.integrity import evaluate_prompt_contamination


class TestEndToEndWorkflow:
    """Test complete workflows from data ingestion to report generation."""
    
    def test_evaluation_pipeline(self):
        """Test complete evaluation pipeline with bug fixes."""
        # Create mock training data
        events_data = []
        for i in range(100):
            events_data.append({
                'step': i,
                'wall_time': i * 0.1,
                'reward_mean': np.random.normal(0.5, 0.1),
                'reward_std': np.random.uniform(0.05, 0.2),
                'kl_mean': np.random.uniform(0.01, 0.1),
                'entropy_mean': np.random.uniform(0.5, 2.0),
                'clip_frac': np.random.uniform(0.0, 0.3),
                'grad_norm': np.random.uniform(0.1, 5.0),
                'lr': 1e-4,
                'loss': np.random.uniform(0.1, 1.0),
            })
        
        df = pd.DataFrame(events_data)
        
        # Test evaluation probes with new fallback logic
        alignment_result = evaluate_alignment(df)
        helpfulness_result = evaluate_helpfulness(df)
        
        # Results should be valid
        assert 'overall_score' in alignment_result
        assert 'overall_score' in helpfulness_result
        assert 0 <= alignment_result['overall_score'] <= 1
        assert 0 <= helpfulness_result['overall_score'] <= 1
        
        # Test statistical calculations
        scores = {
            'reward_mean': df['reward_mean'].mean(),
            'kl_mean': df['kl_mean'].mean(),
            'entropy_mean': df['entropy_mean'].mean(),
        }
        
        # Test with actual data
        sample_data = {
            'reward_mean': df['reward_mean'].values,
            'kl_mean': df['kl_mean'].values,
            'entropy_mean': df['entropy_mean'].values,
        }
        
        ci = calculate_confidence_intervals(scores, sample_size=100, sample_data=sample_data)
        
        # All metrics should have valid confidence intervals
        for metric, (lower, upper) in ci.items():
            assert lower <= scores[metric] <= upper
            assert not np.isnan(lower) and not np.isnan(upper)
    
    def test_monitor_integration(self):
        """Test CheckpointMonitor integration with memory management."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create monitor with custom settings
            monitor = CheckpointMonitor(
                output_dir=temp_dir,
                max_parameter_size_mb=10,  # Small limit for testing
                max_total_memory_mb=50
            )
            
            # Mock model with parameters
            model = Mock()
            small_param = torch.randn(50, 50)  # Small parameter
            large_param = torch.randn(1000, 1000)  # Large parameter
            
            model.named_parameters.return_value = [
                ("small.weight", Mock(data=small_param, requires_grad=True)),
                ("large.weight", Mock(data=large_param, requires_grad=True)),
            ]
            
            # Test weight extraction
            weights = monitor._extract_current_weights(model)
            
            # Should only include small parameter due to size limit
            assert weights is not None
            assert "small.weight" in weights
            assert "large.weight" not in weights
            
            # Test cleanup
            monitor.previous_weights = weights
            monitor._cleanup_weights()
            assert monitor.previous_weights is None
    
    def test_file_processing_integration(self):
        """Test file processing with safety measures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create various log files
            small_log = temp_path / "small.log"
            with open(small_log, 'w') as f:
                f.write("Training started\nNon-deterministic warning\n")
            
            large_log = temp_path / "large.log"
            with open(large_log, 'w') as f:
                f.write("x" * (6 * 1024 * 1024))  # 6MB file
            
            # Test determinism check with file processing
            determinism_result = generate_determinism_card(
                run_dir=temp_path,
                run_id="test_run"
            )
            
            # Should process small file but skip large file
            assert determinism_result is not None
            assert "nondeterminism_hints" in determinism_result


class TestConfigurationIntegration:
    """Test configuration system integration."""
    
    def test_settings_validation(self):
        """Test that settings validation works correctly."""
        # Valid settings should work
        valid_settings = ExtendedConfigSchema()
        assert valid_settings.memory.max_parameter_size_mb == 50
        assert valid_settings.file.max_file_size_mb == 5
        
        # Invalid settings should raise errors
        with pytest.raises(ValueError):
            ExtendedConfigSchema(
                memory=ExtendedConfigSchema().memory.model_copy(
                    update={'max_parameter_size_mb': -1}
                )
            )
        
        with pytest.raises(ValueError):
            ExtendedConfigSchema(
                file=ExtendedConfigSchema().file.model_copy(
                    update={'encoding': 'invalid-encoding'}
                )
            )
    
    def test_environment_variable_loading(self):
        """Test loading settings from environment variables."""
        with patch.dict(os.environ, {
            'RLDK_MAX_PARAMETER_SIZE_MB': '100',
            'RLDK_MAX_FILE_SIZE_MB': '10',
            'RLDK_USE_NEW_STATS': 'true'
        }):
            settings = load_settings_from_env()
            
            assert settings.memory.max_parameter_size_mb == 100
            assert settings.file.max_file_size_mb == 10
            assert settings.bug_fixes.use_new_confidence_intervals is True
            assert settings.bug_fixes.use_new_effect_sizes is True


class TestErrorHandlingIntegration:
    """Test error handling across modules."""
    
    def test_statistical_error_propagation(self):
        """Test that statistical errors are handled consistently."""
        # Test with invalid data
        invalid_scores = {'metric': np.nan}
        
        # Should handle NaN gracefully
        ci = calculate_confidence_intervals(invalid_scores, sample_size=10)
        assert np.isnan(ci['metric'][0]) and np.isnan(ci['metric'][1])
        
        # Test with zero sample size
        ci_zero = calculate_confidence_intervals({'metric': 0.5}, sample_size=0)
        assert ci_zero['metric'] == (0.5, 0.5)
    
    def test_memory_error_handling(self):
        """Test memory error handling in monitor."""
        monitor = CheckpointMonitor()
        
        # Mock model that raises memory error
        model = Mock()
        model.named_parameters.side_effect = RuntimeError("CUDA out of memory")
        
        # Should handle gracefully
        weights = monitor._extract_current_weights(model)
        assert weights is None
    
    def test_file_error_handling(self):
        """Test file error handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create file with permission issues
            restricted_file = temp_path / "restricted.log"
            restricted_file.touch()
            restricted_file.chmod(0o000)  # No permissions
            
            try:
                # Should handle permission errors gracefully
                determinism_result = generate_determinism_card(
                    run_dir=temp_path,
                    run_id="test_run"
                )
                
                # Should still produce result
                assert determinism_result is not None
                
            finally:
                # Restore permissions for cleanup
                restricted_file.chmod(0o644)


class TestFeatureFlagIntegration:
    """Test feature flag integration."""
    
    def test_legacy_method_flag(self):
        """Test legacy method feature flag."""
        scores = {'test': 0.5}
        
        # Test legacy method
        ci_legacy = calculate_confidence_intervals(
            scores, sample_size=100, use_legacy_method=True
        )
        
        # Test new method
        ci_new = calculate_confidence_intervals(
            scores, sample_size=100, use_legacy_method=False
        )
        
        # Both should work but give different results
        assert ci_legacy['test'][0] <= scores['test'] <= ci_legacy['test'][1]
        assert ci_new['test'][0] <= scores['test'] <= ci_new['test'][1]
        
        # Legacy should be more conservative
        legacy_width = ci_legacy['test'][1] - ci_legacy['test'][0]
        new_width = ci_new['test'][1] - ci_new['test'][0]
        assert legacy_width >= new_width * 0.5  # Allow some tolerance
    
    def test_configuration_feature_flags(self):
        """Test configuration-based feature flags."""
        # Enable new methods
        settings = ExtendedConfigSchema()
        settings.bug_fixes.use_new_confidence_intervals = True
        settings.bug_fixes.use_new_effect_sizes = True
        
        # Should be able to get summary
        summary = settings.get_bug_fix_summary()
        assert summary['new_confidence_intervals'] is True
        assert summary['new_effect_sizes'] is True


class TestPerformanceIntegration:
    """Test performance characteristics across modules."""
    
    def test_memory_usage_integration(self):
        """Test memory usage across multiple operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform many operations
        for i in range(50):
            # Statistical calculations
            data = np.random.normal(0.5, 0.2, 1000)
            ci = calculate_confidence_intervals(
                {'metric': np.mean(data)}, 
                sample_size=1000,
                sample_data={'metric': data}
            )
            
            # Monitor operations
            with tempfile.TemporaryDirectory() as temp_dir:
                monitor = CheckpointMonitor(output_dir=temp_dir)
                monitor._cleanup_weights()
        
        final_memory = process.memory_info().rss
        memory_growth = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory growth should be reasonable
        assert memory_growth < 200, f"Excessive memory growth: {memory_growth:.1f}MB"
    
    def test_processing_speed_integration(self):
        """Test processing speed across modules."""
        import time
        
        start_time = time.time()
        
        # Process large dataset
        n_samples = 5000
        data = np.random.normal(0.5, 0.2, n_samples)
        
        # Statistical calculations
        ci = calculate_confidence_intervals(
            {'metric': np.mean(data)}, 
            sample_size=n_samples,
            sample_data={'metric': data}
        )
        
        # Effect size calculation
        baseline_data = np.random.normal(0.3, 0.2, n_samples)
        es = calculate_effect_sizes(
            {'metric': np.mean(data)}, 
            {'metric': np.mean(baseline_data)},
            sample_data={'metric': data},
            baseline_sample_data={'metric': baseline_data}
        )
        
        # File processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            log_file = temp_path / "test.log"
            with open(log_file, 'w') as f:
                f.write("Test log content\n" * 1000)
            
            determinism_result = generate_determinism_card(
                run_dir=temp_path,
                run_id="test_run"
            )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete within reasonable time
        assert total_time < 5.0, f"Too slow: {total_time:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])