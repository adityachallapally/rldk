from __future__ import annotations

import logging

import pandas as pd
import pandas.testing as pdt
import pytest

from rldk.pipelines.ingest import standardize_training_metrics


def test_standardize_schema_coerces_numeric_columns() -> None:
    raw = pd.DataFrame(
        {
            "step": ["1", "2"],
            "reward_mean": ["0.5", "not-a-number"],
            "kl_mean": [0.1, "0.2"],
            "phase": ["train", "train"],
        }
    )

    standardized = standardize_training_metrics(raw)

    assert list(standardized["step"]) == [1, 2]
    assert standardized["reward_mean"].dtype.kind in {"f", "c"}
    assert standardized["kl_mean"].dtype.kind in {"f", "c"}
    assert standardized.loc[standardized["step"] == 1, "reward_mean"].iat[0] == pytest.approx(0.5)
    assert pd.isna(standardized.loc[standardized["step"] == 2, "reward_mean"].iat[0])


def test_standardize_schema_drops_rows_with_invalid_step(caplog: pytest.LogCaptureFixture) -> None:
    raw = pd.DataFrame(
        {
            "step": [1, None, "oops"],
            "reward_mean": [0.5, 0.6, 0.7],
            "phase": ["train", "train", "train"],
        }
    )

    caplog.set_level(logging.WARNING)
    standardized = standardize_training_metrics(raw)

    assert list(standardized["step"]) == [1]
    assert any("Dropped 2 row" in message for message in caplog.messages)


def test_standardize_schema_column_order_and_idempotent() -> None:
    raw = pd.DataFrame(
        {
            "phase": ["train", "train"],
            "custom_metric": ["a", "b"],
            "reward_mean": [0.4, 0.5],
            "step": [2, 1],
        }
    )

    standardized = standardize_training_metrics(raw)

    expected_prefix = [
        "step",
        "phase",
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
        "seed",
        "run_id",
        "git_sha",
    ]
    assert standardized.columns[: len(expected_prefix)].tolist() == expected_prefix
    assert standardized.columns.tolist() == expected_prefix + ["custom_metric"]

    round_trip = standardize_training_metrics(standardized)
    pdt.assert_frame_equal(standardized, round_trip)
