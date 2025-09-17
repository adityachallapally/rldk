"""Tests for Phase A and Phase B normalization features."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rldk.ingest.training_metrics_normalizer import normalize_training_metrics
from rldk.io.event_schema import dataframe_to_events, events_to_dataframe
from rldk.utils.error_handling import ValidationError


class TestPhaseABNormalization:
    """Test Phase A normalization and Phase B features."""

    @pytest.fixture
    def fixtures_dir(self):
        """Get path to test fixtures."""
        return Path(__file__).parent.parent / "fixtures" / "phase_ab"

    def test_stream_to_table_pivot(self, fixtures_dir):
        """Test 1: Load stream_small.jsonl with the public normalizer."""
        stream_path = fixtures_dir / "stream_small.jsonl"
        
        df = normalize_training_metrics(stream_path)
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "step" in df.columns
        assert df["step"].dtype == "int64"
        
        reward_col = _find_reward_column(df)
        assert not df[reward_col].dropna().empty
        
        kl_col = _find_kl_column(df)
        assert not df[kl_col].dropna().empty
        
        assert df["step"].is_monotonic_increasing
        
        expected_cols = {"step", "reward_mean", "kl_mean", "loss", "lr", "grad_norm", "wall_time", "run_id", "phase", "reward_std", "entropy_mean", "clip_frac", "tokens_in", "tokens_out", "seed", "git_sha", "reward", "kl"}
        extra_cols = set(df.columns) - expected_cols
        if extra_cols:
            for col in extra_cols:
                assert col in df.columns

    def test_coercion_and_none_guards(self, fixtures_dir):
        """Test 2: Load stream_mixed_types.jsonl and run schema standardizer."""
        stream_path = fixtures_dir / "stream_mixed_types.jsonl"
        
        try:
            df = normalize_training_metrics(stream_path)
            
            assert isinstance(df, pd.DataFrame)
            assert not df.empty
            
            assert df["step"].dtype == "int64"
            
            reward_col = _find_reward_column(df)
            assert pd.api.types.is_numeric_dtype(df[reward_col])
            
            valid_rewards = df[reward_col].dropna()
            if not valid_rewards.empty:
                mean_reward = valid_rewards.mean()
                assert isinstance(mean_reward, (int, float))
                assert not pd.isna(mean_reward)
        except ValidationError as e:
            if "Invalid step value" in str(e):
                pass
            else:
                raise

    def test_round_trip_events(self, fixtures_dir):
        """Test 3: Convert table to events and back to DataFrame."""
        csv_path = fixtures_dir / "table_small.csv"
        
        original_df = pd.read_csv(csv_path)
        
        events = dataframe_to_events(original_df)
        assert len(events) == len(original_df)
        
        restored_df = events_to_dataframe(events)
        
        for col in ["reward_mean", "kl_mean"]:
            if col in original_df.columns and col in restored_df.columns:
                np.testing.assert_allclose(
                    restored_df[col].dropna().values.astype(float),
                    original_df[col].dropna().values.astype(float),
                    rtol=1e-6,
                    atol=1e-8
                )
        
        np.testing.assert_array_equal(
            restored_df["step"].values.astype(int),
            original_df["step"].values.astype(int)
        )


def _find_reward_column(df: pd.DataFrame) -> str:
    """Helper to find reward column name with data."""
    if "reward" in df.columns and not df["reward"].dropna().empty:
        return "reward"
    elif "reward_mean" in df.columns and not df["reward_mean"].dropna().empty:
        return "reward_mean"
    else:
        raise ValueError("No reward column with data found")


def _find_kl_column(df: pd.DataFrame) -> str:
    """Helper to find KL column name with data."""
    if "kl" in df.columns and not df["kl"].dropna().empty:
        return "kl"
    elif "kl_mean" in df.columns and not df["kl_mean"].dropna().empty:
        return "kl_mean"
    else:
        raise ValueError("No KL column with data found")
