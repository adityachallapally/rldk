"""Tests for metrics-only evaluation suite and column mapping."""

import pandas as pd
import pytest
from rldk.evals.runner import run
from rldk.evals.schema import normalize_columns, get_schema_for_suite, RL_METRICS_SCHEMA


def test_normalize_columns_with_mapping():
    """Test normalize_columns function with user mapping."""
    df = pd.DataFrame({
        "global_step": [1, 2, 3],
        "reward": [0.1, 0.2, 0.15],
        "kl": [0.04, 0.05, 0.06]
    })
    
    column_mapping = {"global_step": "step"}
    result_df, effective_mapping = normalize_columns(df, column_mapping)
    
    assert "step" in result_df.columns
    assert "global_step" not in result_df.columns
    assert effective_mapping == {"global_step": "step"}


def test_normalize_columns_without_mapping():
    """Test normalize_columns function without user mapping."""
    df = pd.DataFrame({
        "step": [1, 2, 3],
        "reward_mean": [0.1, 0.2, 0.15]
    })
    
    result_df, effective_mapping = normalize_columns(df, None)
    
    expected_df = pd.DataFrame({
        "step": [1, 2, 3],
        "reward": [0.1, 0.2, 0.15]
    })
    assert result_df.equals(expected_df)
    assert effective_mapping == {}  # No user mappings applied


def test_get_schema_for_training_metrics():
    """Test that training_metrics suite returns RL_METRICS_SCHEMA."""
    schema = get_schema_for_suite("training_metrics")
    assert schema == RL_METRICS_SCHEMA
    
    # Check required columns
    required_names = [col.name for col in schema.required_columns]
    assert "step" in required_names
    assert "output" not in required_names


def test_training_metrics_suite_minimal_data():
    """Test training_metrics suite with minimal numeric data."""
    df = pd.DataFrame({
        "step": [1, 2, 3, 4],
        "reward_mean": [0.1, 0.2, 0.15, 0.25],
        "kl_mean": [0.04, 0.05, 0.06, 0.05],
        "entropy_mean": [5.2, 5.1, 5.0, 4.9]
    })
    
    result = run(df, suite="training_metrics")
    
    assert result.scores
    assert len(result.scores) > 0
    assert "skipped" in str(result.warnings).lower() or "metrics" in str(result.warnings).lower()


def test_training_metrics_with_column_mapping():
    """Test training_metrics suite with column mapping."""
    df = pd.DataFrame({
        "global_step": [1, 2, 3, 4],
        "reward": [0.1, 0.2, 0.15, 0.25],
        "kl": [0.04, 0.05, 0.06, 0.05],
        "entropy": [5.2, 5.1, 5.0, 4.9]
    })
    
    column_mapping = {"global_step": "step"}
    result = run(df, suite="training_metrics", column_mapping=column_mapping)
    
    assert result.scores
    assert len(result.scores) > 0
    assert result.metadata["effective_column_mapping"] == {"global_step": "step"}


def test_existing_suite_unchanged():
    """Test that existing text-based suites still work unchanged."""
    df = pd.DataFrame({
        "step": [1, 2, 3],
        "output": ["response 1", "response 2", "response 3"],
        "reward_mean": [0.1, 0.2, 0.15]
    })
    
    result = run(df, suite="quick")
    
    warning_text = " ".join(result.warnings).lower()
    assert "metrics only" not in warning_text
    assert "training_metrics" not in warning_text
