"""Tests for first divergence detection."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from rldk.diff.first_divergence import first_divergence, DivergenceReport


class TestFirstDivergence:
    """Test first divergence detection functionality."""
    
    def test_clean_versus_clean_no_divergence(self):
        """Test that two identical runs show no divergence."""
        # Create identical data
        steps = list(range(100))
        data = {
            'step': steps,
            'reward_mean': [0.5 + i * 0.01 for i in steps],
            'kl_mean': [0.1 + i * 0.001 for i in steps],
            'entropy_mean': [0.8 - i * 0.002 for i in steps],
            'lr': [0.001] * len(steps),
            'seed': [42] * len(steps),
            'run_id': ['test_run'] * len(steps),
            'git_sha': ['abc123'] * len(steps)
        }
        
        df_a = pd.DataFrame(data)
        df_b = pd.DataFrame(data)  # Identical to df_a
        
        report = first_divergence(
            df_a, df_b, 
            signals=['reward_mean', 'kl_mean', 'entropy_mean'],
            k_consecutive=3,
            window=20,
            tolerance=2.0
        )
        
        assert not report.diverged
        assert report.first_step is None
        assert len(report.tripped_signals) == 0
        assert len(report.suspected_causes) > 0

    def test_clean_versus_kl_spike_divergence(self):
        """Test that KL spike is detected."""
        # Load fixture data
        fixture_dir = Path(__file__).parent.parent / "runs_fixtures"
        clean_df = pd.read_json(fixture_dir / "clean_ppo.jsonl", lines=True)
        kl_spike_df = pd.read_json(fixture_dir / "kl_spike.jsonl", lines=True)

        report = first_divergence(
            clean_df, kl_spike_df,
            signals=['kl_mean', 'reward_mean'],
            k_consecutive=3,
            window=20,
            tolerance=1.0  # Lower tolerance to detect the spike
        )

        # Should detect divergence due to systematic differences
        assert report.diverged
        # The algorithm finds the first step with k consecutive violations
        assert report.first_step is not None
        assert 'kl_mean' in report.tripped_signals  # KL divergence should be detected
        assert len(report.suspected_causes) > 0
        
        # Check that details contain the divergence event
        assert not report.details.empty
        kl_events = report.details[report.details['signal'] == 'kl_mean']
        assert len(kl_events) > 0
    
    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        # Create minimal data
        df_a = pd.DataFrame({
            'step': [0, 1],
            'reward_mean': [0.5, 0.6],
            'kl_mean': [0.1, 0.11]
        })
        df_b = pd.DataFrame({
            'step': [0, 1],
            'reward_mean': [0.5, 0.6],
            'kl_mean': [0.1, 0.11]
        })
        
        report = first_divergence(
            df_a, df_b,
            signals=['reward_mean', 'kl_mean'],
            k_consecutive=3,
            window=50,  # Window larger than data
            tolerance=2.0
        )
        
        assert not report.diverged
        assert report.first_step is None
        assert "Insufficient common steps" in report.suspected_causes[0]
    
    def test_missing_signals(self):
        """Test handling of missing signals."""
        df_a = pd.DataFrame({
            'step': list(range(50)),
            'reward_mean': [0.5 + i * 0.01 for i in range(50)],
            'kl_mean': [0.1 + i * 0.001 for i in range(50)]
        })
        df_b = pd.DataFrame({
            'step': list(range(50)),
            'reward_mean': [0.5 + i * 0.01 for i in range(50)],
            'kl_mean': [0.1 + i * 0.001 for i in range(50)]
        })
        
        # Test with signal that doesn't exist
        report = first_divergence(
            df_a, df_b,
            signals=['nonexistent_signal'],
            k_consecutive=3,
            window=20,
            tolerance=2.0
        )
        
        assert not report.diverged
        assert len(report.tripped_signals) == 0
    
    def test_k_consecutive_violations(self):
        """Test k-consecutive violation detection."""
        # Create data with a clear spike
        steps = list(range(100))
        base_reward = [0.5 + i * 0.01 for i in steps]
        
        # Add a spike at steps 50-52
        reward_with_spike = base_reward.copy()
        for i in range(50, 53):
            reward_with_spike[i] = base_reward[i] + 0.5  # Large spike
        
        df_a = pd.DataFrame({
            'step': steps,
            'reward_mean': base_reward,
            'kl_mean': [0.1 + i * 0.001 for i in steps]
        })
        df_b = pd.DataFrame({
            'step': steps,
            'reward_mean': reward_with_spike,
            'kl_mean': [0.1 + i * 0.001 for i in steps]
        })
        
        report = first_divergence(
            df_a, df_b,
            signals=['reward_mean'],
            k_consecutive=3,
            window=20,
            tolerance=2.0
        )
        
        assert report.diverged
        assert report.first_step == 52  # End of 3-consecutive violation
        assert 'reward_mean' in report.tripped_signals
    
    def test_tolerance_sensitivity(self):
        """Test that tolerance affects detection."""
        # Create data with small differences
        steps = list(range(100))
        base_reward = [0.5 + i * 0.01 for i in steps]
        reward_with_diff = [r + 0.1 for r in base_reward]  # Small consistent difference
        
        df_a = pd.DataFrame({
            'step': steps,
            'reward_mean': base_reward
        })
        df_b = pd.DataFrame({
            'step': steps,
            'reward_mean': reward_with_diff
        })
        
        # With high tolerance, should not detect
        report_high_tol = first_divergence(
            df_a, df_b,
            signals=['reward_mean'],
            k_consecutive=3,
            window=20,
            tolerance=5.0
        )
        
        # With low tolerance, should detect
        report_low_tol = first_divergence(
            df_a, df_b,
            signals=['reward_mean'],
            k_consecutive=3,
            window=20,
            tolerance=0.5  # Even lower tolerance
        )
        
        assert not report_high_tol.diverged
        assert report_low_tol.diverged
    
    def test_suspected_causes_analysis(self):
        """Test that suspected causes are properly analyzed."""
        # Create data with different seeds
        steps = list(range(50))
        df_a = pd.DataFrame({
            'step': steps,
            'reward_mean': [0.5 + i * 0.01 for i in steps],
            'kl_mean': [0.1 + i * 0.001 for i in steps],
            'lr': [0.001] * len(steps),
            'seed': [42] * len(steps),
            'git_sha': ['abc123'] * len(steps)
        })
        df_b = pd.DataFrame({
            'step': steps,
            'reward_mean': [0.5 + i * 0.01 for i in steps],
            'kl_mean': [0.1 + i * 0.001 for i in steps],
            'lr': [0.001] * len(steps),
            'seed': [43] * len(steps),  # Different seed
            'git_sha': ['def456'] * len(steps)  # Different SHA
        })
        
        report = first_divergence(
            df_a, df_b,
            signals=['reward_mean', 'kl_mean'],
            k_consecutive=3,
            window=20,
            tolerance=2.0
        )
        
        # Should detect different seeds and git SHAs
        causes = ' '.join(report.suspected_causes).lower()
        assert 'different random seeds' in causes
        assert 'different code versions' in causes

    def test_kl_spike_specific_detection(self):
        """Test that the specific KL spike at step 95 is detected when using higher tolerance."""
        # Load fixture data
        fixture_dir = Path(__file__).parent.parent / "runs_fixtures"
        clean_df = pd.read_json(fixture_dir / "clean_ppo.jsonl", lines=True)
        kl_spike_df = pd.read_json(fixture_dir / "kl_spike.jsonl", lines=True)

        # Use parameters that focus on the specific spike at step 95
        report = first_divergence(
            clean_df, kl_spike_df,
            signals=['kl_mean'],  # Focus only on KL
            k_consecutive=2,  # Need 2 consecutive violations to detect the spike
            window=5,  # Very small window to be more sensitive to spikes
            tolerance=0.5  # Low tolerance to detect the spike
        )

        # Should detect divergence due to the specific KL spike
        assert report.diverged
        assert 'kl_mean' in report.tripped_signals
        
        # Check that the spike at step 95 is detected in the details
        kl_events = report.details[report.details['signal'] == 'kl_mean']
        spike_events = kl_events[kl_events['step'] >= 90]
        assert len(spike_events) > 0, "No KL divergence events detected around step 95"
        
        # Check that step 95 is specifically detected
        step_95_events = kl_events[kl_events['step'] == 95]
        assert len(step_95_events) > 0, "KL spike at step 95 not detected"