"""Unit tests for the evaluation length bias metric."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from rldk.evals.metrics.length_bias import evaluate_length_bias


def _build_dataset(correlation: float, count: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    base_lengths = np.linspace(10, 100, count)
    noise = rng.normal(scale=5.0, size=count)
    rewards = base_lengths * correlation + noise
    return pd.DataFrame(
        {
            "step": np.arange(count),
            "response_text": ["x" * int(length) for length in base_lengths],
            "reward_mean": rewards,
        }
    )


def test_evaluate_length_bias_detects_severe_bias() -> None:
    df = _build_dataset(correlation=0.6)
    result = evaluate_length_bias(df, threshold=0.3)

    assert result["passed"] is False
    assert result["severity"] > 0.3
    assert math.isclose(result["score"], 1.0 - result["severity"], rel_tol=1e-6)
    assert result["metrics"]["bias_severity"] == result["severity"]


def test_evaluate_length_bias_handles_low_bias() -> None:
    df = _build_dataset(correlation=0.0)
    result = evaluate_length_bias(df, threshold=0.2)

    assert result["passed"] is True
    assert result["severity"] <= 0.2
    assert result["response_count"] == len(df)
    assert result["num_samples"] == len(df)


def test_evaluate_length_bias_respects_sample_size() -> None:
    df = _build_dataset(correlation=0.4, count=60)
    result = evaluate_length_bias(df, sample_size=15, seed=42)

    assert result["response_count"] == 15
    assert result["num_samples"] <= 15
