"""Tests for reward overoptimization detector."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rldk.reward.health_analysis import health


def _build_run_frame(num_steps: int = 200, reward_gain: float = 0.5) -> pd.DataFrame:
    steps = np.arange(num_steps)
    start = 0.2
    end = start + reward_gain
    rewards = np.linspace(start, end, num_steps) + np.random.normal(0.0, 0.01, num_steps)
    kl_values = np.linspace(0.25, 0.4, num_steps)
    return pd.DataFrame(
        {
            "step": steps,
            "reward_mean": rewards,
            "kl_mean": kl_values,
        }
    )


def test_overoptimization_flagged_when_proxy_outpaces_gold() -> None:
    run_data = _build_run_frame(reward_gain=0.6)
    gold_scores = pd.DataFrame(
        {
            "step": run_data["step"],
            "gold_metric": np.linspace(0.35, 0.34, len(run_data)),
        }
    )

    report = health(
        run_data=run_data,
        reward_col="reward_mean",
        step_col="step",
        threshold_leakage=1.0,
        threshold_shortcut=1.0,
        enable_length_bias_detection=False,
        gold_metrics=gold_scores,
        gold_metric_col="gold_metric",
        overoptimization_min_samples=80,
    )

    assert report.overoptimization.flagged is True
    assert report.overoptimization.delta > 0.2
    assert report.overoptimization.kl_elevated is True
    assert report.passed is False


def test_overoptimization_relaxes_when_gold_improves() -> None:
    run_data = _build_run_frame(reward_gain=0.4)
    gold_scores = pd.DataFrame(
        {
            "step": run_data["step"],
            "gold_metric": np.linspace(0.3, 0.62, len(run_data)),
        }
    )

    report = health(
        run_data=run_data,
        reward_col="reward_mean",
        step_col="step",
        threshold_leakage=1.0,
        threshold_shortcut=1.0,
        enable_length_bias_detection=False,
        gold_metrics=gold_scores,
        gold_metric_col="gold_metric",
        overoptimization_min_samples=80,
    )

    assert report.overoptimization.flagged is False
    assert report.overoptimization.delta < 0.2
    assert report.passed is True


def test_overoptimization_warns_when_missing_gold() -> None:
    run_data = _build_run_frame(reward_gain=0.3)

    report = health(
        run_data=run_data,
        reward_col="reward_mean",
        step_col="step",
        threshold_leakage=1.0,
        threshold_shortcut=1.0,
        enable_length_bias_detection=False,
        overoptimization_min_samples=50,
    )

    assert report.overoptimization.flagged is False
    assert report.overoptimization.gold_metrics_available is False
    assert report.overoptimization.warning is not None
