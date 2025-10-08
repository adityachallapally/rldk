"""Acceptance coverage for Phase A normalization helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_numeric_dtype

from rldk.ingest.stream_normalizer import stream_jsonl_to_dataframe
from rldk.ingest.training_metrics_normalizer import standardize_training_metrics
from rldk.io.event_schema import dataframe_to_events, events_to_dataframe


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "phase_ab"


def _require_column(df: pd.DataFrame, options: list[str]) -> str:
    for name in options:
        if name in df.columns:
            return name
    pytest.fail(f"None of the expected columns {options} are present: {df.columns}")


def test_stream_to_table_pivot() -> None:
    path = FIXTURES / "stream_small.jsonl"

    df = stream_jsonl_to_dataframe(path)

    assert not df.empty
    assert df["step"].is_monotonic_increasing
    assert is_numeric_dtype(df["step"])

    reward_col = _require_column(df, ["reward_mean", "reward"])
    kl_col = _require_column(df, ["kl_mean", "kl"])
    assert is_numeric_dtype(df[reward_col])
    assert is_numeric_dtype(df[kl_col])

    # ensure passthrough for additional metrics
    assert any(column not in {"step", reward_col, kl_col} for column in df.columns)


def test_coercion_and_none_guards(tmp_path: Path) -> None:
    path = FIXTURES / "stream_mixed_types.jsonl"

    raw_df = stream_jsonl_to_dataframe(path)
    reward_col = _require_column(raw_df, ["reward_mean", "reward"])

    # introduce mixed typing and an invalid step row
    mixed = raw_df.copy()
    mixed[reward_col] = mixed[reward_col].astype("string")
    if "kl_mean" in mixed.columns:
        mixed["kl_mean"] = mixed["kl_mean"].astype("string")
    invalid_row = {column: None for column in mixed.columns}
    invalid_row["step"] = "not-a-number"
    invalid_row[reward_col] = "0.123"
    mixed = pd.concat([mixed, pd.DataFrame([invalid_row])], ignore_index=True)

    standardized = standardize_training_metrics(mixed)

    canonical_reward = _require_column(standardized, ["reward_mean", "reward"])
    assert is_numeric_dtype(standardized[canonical_reward])
    if "kl_mean" in standardized.columns:
        assert is_numeric_dtype(standardized["kl_mean"])

    # invalid row should have been dropped
    assert len(standardized) == len(raw_df)

    reward_mean = standardized[canonical_reward].dropna().mean()
    assert np.isfinite(reward_mean)


def test_round_trip_events(tmp_path: Path) -> None:
    table_path = FIXTURES / "table_small.csv"
    table_df = pd.read_csv(table_path)

    events = dataframe_to_events(table_df)
    # inject malformed events that should be skipped silently
    events.append({"step": "bad", "metrics": {"reward_mean": "oops"}})
    events.append("not-an-event")  # type: ignore[arg-type]

    round_trip_df = events_to_dataframe(events)

    reward_col = _require_column(round_trip_df, ["reward_mean", "reward"])
    kl_col = _require_column(round_trip_df, ["kl_mean", "kl"])

    merged = round_trip_df.merge(table_df, on="step", suffixes=("_actual", "_expected"))
    assert not merged.empty

    np.testing.assert_allclose(
        merged[f"{reward_col}_actual"],
        merged[f"{reward_col}_expected"],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        merged[f"{kl_col}_actual"],
        merged[f"{kl_col}_expected"],
        rtol=1e-6,
        atol=1e-6,
    )
