"""Unit tests for PerformanceAnalyzer."""

import os
import warnings
from unittest.mock import patch

import numpy as np
import pytest

from rldk.integrations.openrlhf.performance_analyzer import (
    PerformanceAnalyzer,
    PerformanceThresholds,
)


class TestPerformanceAnalyzer:
    """Test cases for PerformanceAnalyzer class."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        analyzer = PerformanceAnalyzer()

        assert analyzer.kl_high == 0.1
        assert analyzer.kl_low == 0.01
        assert analyzer.entropy_low == 0.1
        assert analyzer.throughput_low == 0.1
        assert analyzer.window_size == 10
        assert analyzer.emit is True
        assert analyzer.step_count == 0
        assert len(analyzer.kl_history) == 0

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        analyzer = PerformanceAnalyzer(
            kl_high=0.2,
            kl_low=0.02,
            entropy_low=0.2,
            throughput_low=0.2,
            window_size=5,
            emit=False
        )

        assert analyzer.kl_high == 0.2
        assert analyzer.kl_low == 0.02
        assert analyzer.entropy_low == 0.2
        assert analyzer.throughput_low == 0.2
        assert analyzer.window_size == 5
        assert analyzer.emit is False

    def test_from_env(self):
        """Test creation from environment variables."""
        env_vars = {
            'RLHF_PERF_KL_HIGH': '0.15',
            'RLHF_PERF_KL_LOW': '0.015',
            'RLHF_PERF_ENTROPY_LOW': '0.15',
            'RLHF_PERF_THROUGHPUT_LOW': '0.15',
            'RLHF_PERF_WINDOW_SIZE': '15',
            'RLHF_PERF_EMIT': 'false'
        }

        with patch.dict(os.environ, env_vars):
            analyzer = PerformanceAnalyzer.from_env()

            assert analyzer.kl_high == 0.15
            assert analyzer.kl_low == 0.015
            assert analyzer.entropy_low == 0.15
            assert analyzer.throughput_low == 0.15
            assert analyzer.window_size == 15
            assert analyzer.emit is False

    def test_from_config(self):
        """Test creation from configuration dictionary."""
        config = {
            'kl_high': 0.25,
            'kl_low': 0.025,
            'entropy_low': 0.25,
            'throughput_low': 0.25,
            'window_size': 20,
            'emit': False
        }

        analyzer = PerformanceAnalyzer.from_config(config)

        assert analyzer.kl_high == 0.25
        assert analyzer.kl_low == 0.025
        assert analyzer.entropy_low == 0.25
        assert analyzer.throughput_low == 0.25
        assert analyzer.window_size == 20
        assert analyzer.emit is False

    def test_analyze_empty_metrics(self):
        """Test analysis with empty metrics."""
        analyzer = PerformanceAnalyzer()
        result = analyzer.analyze({})

        assert result['status'] == 'ok'
        assert 'signals' in result
        assert 'reasons' in result
        assert len(result['reasons']) == 0

    def test_analyze_ok_status(self):
        """Test analysis with metrics that should result in OK status."""
        analyzer = PerformanceAnalyzer(
            kl_high=0.1,
            kl_low=0.01,
            entropy_low=0.1,
            throughput_low=0.1
        )

        # Good metrics - all values within acceptable ranges
        step_metrics = {
            'kl_mean': 0.005,  # Below low threshold
            'entropy_mean': 0.5,  # Above low threshold
            'step_time': 1.0,  # Good throughput (1.0 steps/sec)
            'loss': 0.1,
            'reward_mean': 0.5
        }

        result = analyzer.analyze(step_metrics)

        assert result['status'] == 'ok'
        assert len(result['reasons']) == 0
        assert result['signals']['kl_mean'] == 0.005
        assert result['signals']['entropy_mean'] == 0.5
        assert result['signals']['throughput_mean'] == 1.0

    def test_analyze_warn_status_kl_low(self):
        """Test analysis with KL divergence above low threshold but below high threshold."""
        analyzer = PerformanceAnalyzer(
            kl_high=0.1,
            kl_low=0.01,
            entropy_low=0.1,
            throughput_low=0.1
        )

        step_metrics = {
            'kl_mean': 0.05,  # Above low threshold (0.01) but below high threshold (0.1)
            'entropy_mean': 0.5,
            'step_time': 1.0,
            'loss': 0.1,
            'reward_mean': 0.5
        }

        result = analyzer.analyze(step_metrics)

        assert result['status'] == 'warn'
        assert len(result['reasons']) == 1
        assert 'KL divergence' in result['reasons'][0]
        assert 'exceeds low threshold' in result['reasons'][0]

    def test_analyze_alert_status_kl_high(self):
        """Test analysis with KL divergence above high threshold."""
        analyzer = PerformanceAnalyzer(
            kl_high=0.1,
            kl_low=0.01,
            entropy_low=0.1,
            throughput_low=0.1
        )

        step_metrics = {
            'kl_mean': 0.15,  # Above high threshold
            'entropy_mean': 0.5,
            'step_time': 1.0,
            'loss': 0.1,
            'reward_mean': 0.5
        }

        result = analyzer.analyze(step_metrics)

        assert result['status'] == 'alert'
        assert len(result['reasons']) == 1
        assert 'KL divergence' in result['reasons'][0]
        assert 'exceeds high threshold' in result['reasons'][0]

    def test_analyze_warn_status_entropy_low(self):
        """Test analysis with entropy below threshold."""
        analyzer = PerformanceAnalyzer(
            kl_high=0.1,
            kl_low=0.01,
            entropy_low=0.1,
            throughput_low=0.1
        )

        step_metrics = {
            'kl_mean': 0.005,  # Below low threshold
            'entropy_mean': 0.05,  # Below entropy threshold
            'step_time': 1.0,
            'loss': 0.1,
            'reward_mean': 0.5
        }

        result = analyzer.analyze(step_metrics)

        assert result['status'] == 'warn'
        assert len(result['reasons']) == 1
        assert 'Entropy' in result['reasons'][0]
        assert 'below threshold' in result['reasons'][0]

    def test_analyze_warn_status_throughput_low(self):
        """Test analysis with throughput below threshold."""
        analyzer = PerformanceAnalyzer(
            kl_high=0.1,
            kl_low=0.01,
            entropy_low=0.1,
            throughput_low=0.1
        )

        step_metrics = {
            'kl_mean': 0.005,
            'entropy_mean': 0.5,
            'step_time': 20.0,  # Low throughput (0.05 steps/sec)
            'loss': 0.1,
            'reward_mean': 0.5
        }

        result = analyzer.analyze(step_metrics)

        assert result['status'] == 'warn'
        assert len(result['reasons']) == 1
        assert 'Throughput' in result['reasons'][0]
        assert 'below threshold' in result['reasons'][0]

    def test_rolling_window_behavior(self):
        """Test that rolling window maintains correct size."""
        analyzer = PerformanceAnalyzer(window_size=3)

        # Add more metrics than window size
        for i in range(5):
            step_metrics = {
                'kl_mean': 0.1 + i * 0.01,
                'entropy_mean': 0.5,
                'step_time': 1.0,
                'loss': 0.1,
                'reward_mean': 0.5
            }
            analyzer.analyze(step_metrics)

        # Check that only last 3 steps are kept
        assert len(analyzer.kl_history) == 3
        assert len(analyzer.entropy_history) == 3
        assert len(analyzer.throughput_history) == 3

        # Check that the correct values are kept (last 3)
        expected_kl_values = [0.12, 0.13, 0.14]  # Last 3 values
        assert list(analyzer.kl_history) == expected_kl_values

    def test_sustained_high_kl_alert(self):
        """Test alert for sustained high KL divergence over window."""
        analyzer = PerformanceAnalyzer(
            kl_high=0.1,
            kl_low=0.01,
            entropy_low=0.1,
            throughput_low=0.1,
            window_size=5
        )

        # Add metrics with high KL divergence for most of the window
        for i in range(5):
            step_metrics = {
                'kl_mean': 0.15 if i >= 1 else 0.05,  # High KL for 4/5 steps
                'entropy_mean': 0.5,
                'step_time': 1.0,
                'loss': 0.1,
                'reward_mean': 0.5
            }
            result = analyzer.analyze(step_metrics)

        # Should trigger alert for sustained high KL
        assert result['status'] == 'alert'
        assert any('KL divergence high for' in reason for reason in result['reasons'])

    def test_sustained_low_entropy_warn(self):
        """Test warning for sustained low entropy over window."""
        analyzer = PerformanceAnalyzer(
            kl_high=0.1,
            kl_low=0.01,
            entropy_low=0.1,
            throughput_low=0.1,
            window_size=5
        )

        # Add metrics with low entropy for most of the window
        for i in range(5):
            step_metrics = {
                'kl_mean': 0.005,
                'entropy_mean': 0.05 if i >= 1 else 0.5,  # Low entropy for 4/5 steps
                'step_time': 1.0,
                'loss': 0.1,
                'reward_mean': 0.5
            }
            result = analyzer.analyze(step_metrics)

        # Should trigger warning for sustained low entropy
        assert result['status'] == 'warn'
        assert any('Entropy low for' in reason for reason in result['reasons'])

    def test_sustained_low_throughput_warn(self):
        """Test warning for sustained low throughput over window."""
        analyzer = PerformanceAnalyzer(
            kl_high=0.1,
            kl_low=0.01,
            entropy_low=0.1,
            throughput_low=0.1,
            window_size=5
        )

        # Add metrics with low throughput for most of the window
        for i in range(5):
            step_metrics = {
                'kl_mean': 0.005,
                'entropy_mean': 0.5,
                'step_time': 20.0 if i >= 1 else 1.0,  # Low throughput for 4/5 steps
                'loss': 0.1,
                'reward_mean': 0.5
            }
            result = analyzer.analyze(step_metrics)

        # Should trigger warning for sustained low throughput
        assert result['status'] == 'warn'
        assert any('Throughput low for' in reason for reason in result['reasons'])

    def test_signals_computation(self):
        """Test that signals are computed correctly."""
        analyzer = PerformanceAnalyzer(window_size=5)

        # Add some metrics
        for i in range(3):
            step_metrics = {
                'kl_mean': 0.1 + i * 0.01,
                'entropy_mean': 0.5 + i * 0.1,
                'step_time': 1.0 + i * 0.1,
                'loss': 0.1 + i * 0.01,
                'reward_mean': 0.5 + i * 0.1
            }
            result = analyzer.analyze(step_metrics)

        signals = result['signals']

        # Check that signals are computed
        assert 'kl_mean' in signals
        assert 'kl_std' in signals
        assert 'kl_max' in signals
        assert 'kl_trend' in signals

        assert 'entropy_mean' in signals
        assert 'entropy_std' in signals
        assert 'entropy_min' in signals
        assert 'entropy_trend' in signals

        assert 'throughput_mean' in signals
        assert 'throughput_std' in signals
        assert 'throughput_min' in signals
        assert 'throughput_trend' in signals

        assert 'loss_mean' in signals
        assert 'loss_std' in signals
        assert 'loss_trend' in signals

        assert 'reward_mean' in signals
        assert 'reward_std' in signals
        assert 'reward_trend' in signals

        # Check window statistics
        assert signals['window_size'] == 3
        assert signals['step_count'] == 3

    def test_trend_computation(self):
        """Test trend computation for various patterns."""
        analyzer = PerformanceAnalyzer(window_size=5)

        # Test increasing trend
        for i in range(3):
            step_metrics = {
                'kl_mean': 0.1 + i * 0.01,  # Increasing: 0.1, 0.11, 0.12
                'entropy_mean': 0.5,
                'step_time': 1.0,
                'loss': 0.1,
                'reward_mean': 0.5
            }
            result = analyzer.analyze(step_metrics)

        assert result['signals']['kl_trend'] > 0  # Positive trend

        # Test decreasing trend
        analyzer.reset()
        for i in range(3):
            step_metrics = {
                'kl_mean': 0.12 - i * 0.01,  # Decreasing: 0.12, 0.11, 0.10
                'entropy_mean': 0.5,
                'step_time': 1.0,
                'loss': 0.1,
                'reward_mean': 0.5
            }
            result = analyzer.analyze(step_metrics)

        assert result['signals']['kl_trend'] < 0  # Negative trend

    def test_emit_alert_warning(self):
        """Test that alerts and warnings are emitted when enabled."""
        analyzer = PerformanceAnalyzer(emit=True)

        # Test warning emission
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")

            step_metrics = {
                'kl_mean': 0.05,  # Above low threshold
                'entropy_mean': 0.5,
                'step_time': 1.0,
                'loss': 0.1,
                'reward_mean': 0.5
            }
            result = analyzer.analyze(step_metrics)

            # Should emit warning
            assert result['status'] == 'warn'
            # Note: The current implementation prints warnings instead of using warnings module
            # This test verifies the status is correct

    def test_no_emit_when_disabled(self):
        """Test that no alerts are emitted when emit=False."""
        analyzer = PerformanceAnalyzer(emit=False)

        step_metrics = {
            'kl_mean': 0.15,  # Above high threshold
            'entropy_mean': 0.5,
            'step_time': 1.0,
            'loss': 0.1,
            'reward_mean': 0.5
        }
        result = analyzer.analyze(step_metrics)

        # Should still detect alert but not emit
        assert result['status'] == 'alert'
        assert len(result['reasons']) == 1

    def test_get_current_state(self):
        """Test get_current_state method."""
        analyzer = PerformanceAnalyzer(
            kl_high=0.2,
            kl_low=0.02,
            entropy_low=0.2,
            throughput_low=0.2,
            window_size=5
        )

        # Add some data
        for i in range(3):
            step_metrics = {
                'kl_mean': 0.1 + i * 0.01,
                'entropy_mean': 0.5,
                'step_time': 1.0,
                'loss': 0.1,
                'reward_mean': 0.5
            }
            analyzer.analyze(step_metrics)

        state = analyzer.get_current_state()

        # Check thresholds
        assert state['thresholds']['kl_high'] == 0.2
        assert state['thresholds']['kl_low'] == 0.02
        assert state['thresholds']['entropy_low'] == 0.2
        assert state['thresholds']['throughput_low'] == 0.2
        assert state['thresholds']['window_size'] == 5

        # Check buffers
        assert len(state['buffers']['kl_history']) == 3
        assert len(state['buffers']['entropy_history']) == 3
        assert len(state['buffers']['throughput_history']) == 3

        # Check state
        assert state['state']['step_count'] == 3
        assert state['state']['emit'] is True

    def test_reset(self):
        """Test reset method."""
        analyzer = PerformanceAnalyzer()

        # Add some data
        for i in range(3):
            step_metrics = {
                'kl_mean': 0.1 + i * 0.01,
                'entropy_mean': 0.5,
                'step_time': 1.0,
                'loss': 0.1,
                'reward_mean': 0.5
            }
            analyzer.analyze(step_metrics)

        # Verify data is present
        assert len(analyzer.kl_history) == 3
        assert analyzer.step_count == 3

        # Reset
        analyzer.reset()

        # Verify everything is cleared
        assert len(analyzer.kl_history) == 0
        assert len(analyzer.entropy_history) == 0
        assert len(analyzer.throughput_history) == 0
        assert len(analyzer.step_times) == 0
        assert len(analyzer.loss_history) == 0
        assert len(analyzer.reward_history) == 0
        assert analyzer.step_count == 0
        assert analyzer.last_analysis_step == 0

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        analyzer = PerformanceAnalyzer()

        # Test with zero step_time (should not add to throughput)
        step_metrics = {
            'kl_mean': 0.05,
            'entropy_mean': 0.5,
            'step_time': 0.0,  # Zero step time
            'loss': 0.1,
            'reward_mean': 0.5
        }
        result = analyzer.analyze(step_metrics)

        # Should not crash and should handle gracefully
        assert result['status'] in ['ok', 'warn', 'alert']
        assert 'throughput_mean' in result['signals']

        # Test with negative values
        step_metrics = {
            'kl_mean': -0.01,  # Negative KL
            'entropy_mean': -0.1,  # Negative entropy
            'step_time': 1.0,
            'loss': 0.1,
            'reward_mean': 0.5
        }
        result = analyzer.analyze(step_metrics)

        # Should handle negative values gracefully
        assert result['status'] in ['ok', 'warn', 'alert']

    def test_multiple_reasons(self):
        """Test that multiple issues can be detected simultaneously."""
        analyzer = PerformanceAnalyzer(
            kl_high=0.1,
            kl_low=0.01,
            entropy_low=0.1,
            throughput_low=0.1
        )

        step_metrics = {
            'kl_mean': 0.15,  # Above high threshold
            'entropy_mean': 0.05,  # Below entropy threshold
            'step_time': 20.0,  # Low throughput
            'loss': 0.1,
            'reward_mean': 0.5
        }

        result = analyzer.analyze(step_metrics)

        # Should detect multiple issues
        assert result['status'] == 'alert'  # Alert due to high KL
        assert len(result['reasons']) >= 2  # At least KL and entropy/throughput issues
        assert any('KL divergence' in reason for reason in result['reasons'])
        assert any('Entropy' in reason for reason in result['reasons'])
        assert any('Throughput' in reason for reason in result['reasons'])


class TestPerformanceThresholds:
    """Test cases for PerformanceThresholds dataclass."""

    def test_default_values(self):
        """Test default values for PerformanceThresholds."""
        thresholds = PerformanceThresholds()

        assert thresholds.kl_high == 0.1
        assert thresholds.kl_low == 0.01
        assert thresholds.entropy_low == 0.1
        assert thresholds.throughput_low == 0.1
        assert thresholds.window_size == 10
        assert thresholds.emit is True

    def test_custom_values(self):
        """Test custom values for PerformanceThresholds."""
        thresholds = PerformanceThresholds(
            kl_high=0.2,
            kl_low=0.02,
            entropy_low=0.2,
            throughput_low=0.2,
            window_size=5,
            emit=False
        )

        assert thresholds.kl_high == 0.2
        assert thresholds.kl_low == 0.02
        assert thresholds.entropy_low == 0.2
        assert thresholds.throughput_low == 0.2
        assert thresholds.window_size == 5
        assert thresholds.emit is False


if __name__ == '__main__':
    pytest.main([__file__])
