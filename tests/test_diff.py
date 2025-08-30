"""Tests for the diff module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from rldk.diff import first_divergence, DivergenceReport


class TestFirstDivergence:
    """Test divergence detection functionality."""
    
    def test_no_divergence(self):
        """Test when runs are identical."""
        # Create identical dataframes
        steps = list(range(100))
        df_a = pd.DataFrame({
            'step': steps,
            'reward_mean': [0.5 + 0.01 * i for i in steps],
            'kl_mean': [0.1 + 0.001 * i for i in steps]
        })
        
        df_b = df_a.copy()
        
        report = first_divergence(df_a, df_b, ['reward_mean', 'kl_mean'])
        
        assert not report.diverged
        assert report.first_step is None
        assert len(report.tripped_signals) == 0
    
    def test_kl_spike_detection(self):
        """Test detection of injected KL spike."""
        # Create baseline data
        steps = list(range(200))
        baseline_reward = [0.5 + 0.01 * i for i in steps]
        baseline_kl = [0.1 + 0.001 * i for i in steps]
        
        df_a = pd.DataFrame({
            'step': steps,
            'reward_mean': baseline_reward,
            'kl_mean': baseline_kl
        })
        
        # Create data with KL spike around step 100
        df_b = df_a.copy()
        spike_start, spike_end = 95, 105
        for i in range(spike_start, spike_end):
            df_b.loc[df_b['step'] == i, 'kl_mean'] *= 3.0  # 3x spike
        
        report = first_divergence(df_a, df_b, ['kl_mean'], k_consecutive=3, window=20)
        
        # Should detect divergence around the spike
        assert report.diverged
        assert report.first_step is not None
        assert report.first_step >= spike_start
        assert 'kl_mean' in report.tripped_signals
    
    def test_insufficient_steps(self):
        """Test handling of insufficient common steps."""
        df_a = pd.DataFrame({'step': [1, 2], 'reward_mean': [0.5, 0.6]})
        df_b = pd.DataFrame({'step': [1, 2], 'reward_mean': [0.5, 0.6]})
        
        report = first_divergence(df_a, df_b, ['reward_mean'], window=50)
        
        assert not report.diverged
        assert "Insufficient common steps" in report.notes[0]
    
    def test_missing_metrics(self):
        """Test handling of missing metrics."""
        steps = list(range(50))
        df_a = pd.DataFrame({
            'step': steps,
            'reward_mean': [0.5 + 0.01 * i for i in steps]
        })
        
        df_b = pd.DataFrame({
            'step': steps,
            'reward_mean': [0.5 + 0.01 * i for i in steps]
        })
        
        # Try to monitor a metric that doesn't exist
        report = first_divergence(df_a, df_b, ['nonexistent_metric'])
        
        # Should not crash, but no divergence detected
        assert not report.diverged


class TestDivergenceReport:
    """Test DivergenceReport dataclass."""
    
    def test_report_creation(self):
        """Test creating a divergence report."""
        report = DivergenceReport(
            diverged=True,
            first_step=100,
            tripped_signals=['kl_mean'],
            notes=['Test note'],
            report_path='test_report.md',
            events_csv_path='test_events.csv'
        )
        
        assert report.diverged
        assert report.first_step == 100
        assert 'kl_mean' in report.tripped_signals
        assert 'Test note' in report.notes
        assert report.report_path == 'test_report.md'
        assert report.events_csv_path == 'test_events.csv'
