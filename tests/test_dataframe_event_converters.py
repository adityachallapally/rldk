"""Tests for converting between TrainingMetrics DataFrames and normalized events."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from rldk.io.event_schema import dataframe_to_events, events_to_dataframe


def test_dataframe_event_round_trip_preserves_metrics() -> None:
    df = pd.DataFrame(
        {
            "step": [0, 1],
            "reward_mean": [1.0, 1.5],
            "reward_std": [0.1, 0.2],
            "kl_mean": [0.01, 0.015],
            "entropy_mean": [2.0, 1.9],
            "clip_frac": [0.1, 0.12],
            "grad_norm": [0.9, 1.1],
            "lr": [3e-4, 2.5e-4],
            "loss": [0.5, 0.45],
            "tokens_in": [128, 128],
            "tokens_out": [120, 130],
            "wall_time": [1_690_000_000.0, 1_690_000_100.0],
            "run_id": ["run-123", "run-123"],
            "seed": [42, 42],
            "phase": ["train", "train"],
        }
    )

    events = dataframe_to_events(df)
    assert len(events) == len(df)

    round_trip = events_to_dataframe(events)

    numeric_columns = [
        "reward_mean",
        "reward_std",
        "kl_mean",
        "entropy_mean",
        "clip_frac",
        "grad_norm",
        "lr",
        "loss",
        "tokens_in",
        "tokens_out",
        "wall_time",
    ]

    for column in numeric_columns:
        np.testing.assert_allclose(
            round_trip[column].to_numpy(dtype=float),
            df[column].to_numpy(dtype=float),
            rtol=1e-6,
            atol=1e-8,
            equal_nan=True,
        )

    np.testing.assert_array_equal(round_trip["step"].to_numpy(), df["step"].to_numpy())
    assert list(round_trip["run_id"].fillna("")) == list(df["run_id"].fillna(""))
    assert list(round_trip["seed"].fillna(pd.NA)) == list(df["seed"].fillna(pd.NA))
    assert list(round_trip["phase"].fillna("")) == list(df["phase"].fillna(""))


def test_events_to_dataframe_handles_non_numeric_metrics() -> None:
    bad_event = {
        "step": 3,
        "wall_time": 100.0,
        "metrics": {"reward_mean": "not-a-number", "kl_mean": "0.2"},
        "rng": {"seed": 11},
        "data_slice": {},
        "model_info": {"run_id": "run-456", "phase": "eval"},
    }

    df = events_to_dataframe([bad_event])
    assert math.isnan(df.loc[0, "reward_mean"])
    assert df.loc[0, "kl_mean"] == pytest.approx(0.2)
    assert df.loc[0, "run_id"] == "run-456"
    assert df.loc[0, "phase"] == "eval"
    assert df.loc[0, "seed"] == 11


def test_dataframe_to_events_preserves_network_metrics() -> None:
    df = pd.DataFrame(
        {
            "step": [5],
            "network_bandwidth": [250.0],
            "latency_ms": [12.5],
        }
    )

    events = dataframe_to_events(df, run_id="net-run")
    assert len(events) == 1
    event_metrics = events[0].metrics
    assert event_metrics["network_bandwidth"] == pytest.approx(250.0)
    assert event_metrics["latency_ms"] == pytest.approx(12.5)

    restored = events_to_dataframe(events)
    assert restored.loc[0, "network_bandwidth"] == pytest.approx(250.0)
    assert restored.loc[0, "latency_ms"] == pytest.approx(12.5)


def test_dataframe_to_events_ignores_missing_metrics() -> None:
    df = pd.DataFrame(
        {
            "step": [0, 1],
            "reward_mean": [None, 0.75],
            "kl_mean": [0.12, float("nan")],
        }
    )

    events = dataframe_to_events(df, run_id="missing-metrics")

    assert len(events) == 2
    assert "reward_mean" not in events[0].metrics
    assert events[1].metrics["reward_mean"] == pytest.approx(0.75)
    assert events[0].metrics["kl_mean"] == pytest.approx(0.12)
    assert "kl_mean" not in events[1].metrics


def test_dataframe_to_events_orders_by_step_stably() -> None:
    df = pd.DataFrame(
        {
            "step": [3, 1, 2, 1],
            "reward_mean": [0.3, 0.1, 0.2, 0.15],
            "wall_time": [30.0, 10.0, 20.0, 11.0],
        }
    )

    events = dataframe_to_events(df, run_id="ordering")

    assert [event.step for event in events] == [1, 1, 2, 3]
    assert [event.metrics["reward_mean"] for event in events] == [0.1, 0.15, 0.2, 0.3]
