"""Unit tests for the catastrophic forgetting evaluation metric."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rldk.evals.metrics.catastrophic_forgetting import evaluate_catastrophic_forgetting


def _make_dataframe(scores_a, scores_b):
    rows = []
    for idx, score in enumerate(scores_a):
        rows.append({"step": idx, "output": f"task-a-{idx}", "task": "task_a", "score": score})
    for idx, score in enumerate(scores_b):
        rows.append({
            "step": len(scores_a) + idx,
            "output": f"task-b-{idx}",
            "task": "task_b",
            "score": score,
        })
    return pd.DataFrame(rows)


def test_no_regression_scores_maximize_component_score():
    current = _make_dataframe(
        scores_a=np.full(6, 0.82, dtype=float),
        scores_b=np.full(6, 0.78, dtype=float),
    )

    baselines = {
        "task_a": {"mean": 0.8, "std": 0.02, "count": 30},
        "task_b": {"mean": 0.75, "std": 0.03, "count": 24},
    }

    result = evaluate_catastrophic_forgetting(current, baseline_summaries=baselines)

    assert result["score"] == 1.0
    assert result["regressed_tasks"] == []
    assert len(result["stable_tasks"]) == 2


def test_mild_regression_produces_intermediate_score():
    current = _make_dataframe(
        scores_a=np.full(6, 0.72, dtype=float),
        scores_b=np.full(6, 0.76, dtype=float),
    )

    baselines = {
        "task_a": {"mean": 0.75, "std": 0.05, "count": 40},
        "task_b": {"mean": 0.74, "std": 0.05, "count": 35},
    }

    result = evaluate_catastrophic_forgetting(current, baseline_summaries=baselines)

    assert 0.0 < result["score"] < 1.0
    assert result["regressed_tasks"] == []
    assert {task["task"] for task in result["stable_tasks"]} == {"task_a", "task_b"}


def test_catastrophic_regression_flags_tasks_and_zero_score():
    current = _make_dataframe(
        scores_a=np.full(6, 0.6, dtype=float),
        scores_b=np.full(6, 0.77, dtype=float),
    )

    baselines = {
        "task_a": {"mean": 0.75, "std": 0.02, "count": 90},
        "task_b": {"mean": 0.74, "std": 0.05, "count": 10},
    }

    result = evaluate_catastrophic_forgetting(current, baseline_summaries=baselines)

    assert result["score"] < 0.5
    assert any(task["task"] == "task_a" for task in result["regressed_tasks"])
    assert all(task["task"] != "task_a" or task["delta_mean"] < 0 for task in result["regressed_tasks"])


def test_missing_baselines_reported_gracefully():
    current = _make_dataframe(
        scores_a=np.full(4, 0.8, dtype=float),
        scores_b=np.full(4, 0.81, dtype=float),
    )

    baselines = {"task_a": {"mean": 0.79, "std": 0.02, "count": 20}}

    result = evaluate_catastrophic_forgetting(current, baseline_summaries=baselines)

    assert np.isnan(result["score"])
    assert result["missing_baselines"] == ["task_b"]
    assert "task_b" in result["recommendations"][-1]


def test_baseline_dataframe_with_nan_mean_is_skipped():
    current = _make_dataframe(
        scores_a=np.full(4, 0.82, dtype=float),
        scores_b=np.full(4, 0.8, dtype=float),
    )

    baseline_df = pd.DataFrame(
        {
            "task": ["task_a", "task_b"],
            "mean": [np.nan, 0.78],
            "std": [0.02, 0.03],
            "count": [30, 25],
        }
    )

    result = evaluate_catastrophic_forgetting(current, baseline_summaries=baseline_df)

    assert "task_a" in result["missing_baselines"]
    assert any("mean value is missing" in warning for warning in result["warnings"])
    assert result["metadata"]["baseline_available"]


def test_zero_regression_threshold_falls_back_with_warning():
    current = _make_dataframe(
        scores_a=np.full(6, 0.82, dtype=float),
        scores_b=np.full(6, 0.78, dtype=float),
    )

    baselines = {
        "task_a": {"mean": 0.8, "std": 0.02, "count": 30},
        "task_b": {"mean": 0.75, "std": 0.03, "count": 24},
    }

    result = evaluate_catastrophic_forgetting(
        current,
        baseline_summaries=baselines,
        regression_threshold=0.0,
    )

    assert any("Regression threshold of 0.0" in warning for warning in result["warnings"])
    assert result["metadata"]["regression_threshold"] < 0
