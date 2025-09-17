from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from rldk.ingest import stream_jsonl_to_dataframe
from rldk.utils.error_handling import ValidationError


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_stream_jsonl_to_dataframe_basic(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    _write_jsonl(
        path,
        [
            {"time": 1.0, "step": 1, "name": "reward_mean", "value": 1.5, "run_id": "run-a"},
            {"time": 1.0, "step": 1, "name": "kl_mean", "value": 0.1},
            {"time": 2.0, "step": 1, "name": "loss", "value": 0.25},
            {"time": 3.0, "step": 2, "name": "lr", "value": 3e-4},
            {"time": 4.0, "step": 2, "name": "grad_norm", "value": 0.45},
        ],
    )

    df = stream_jsonl_to_dataframe(path)

    assert list(df["step"]) == [1, 2]
    assert "reward_mean" in df.columns
    assert "kl_mean" in df.columns
    assert "loss" in df.columns
    assert "lr" in df.columns
    assert "grad_norm" in df.columns
    assert df.loc[df["step"] == 1, "reward_mean"].iat[0] == pytest.approx(1.5)
    assert df.loc[df["step"] == 1, "kl_mean"].iat[0] == pytest.approx(0.1)
    assert df.loc[df["step"] == 2, "grad_norm"].iat[0] == pytest.approx(0.45)
    assert df.loc[df["step"] == 1, "run_id"].iat[0] == "run-a"


def test_stream_jsonl_to_dataframe_with_extra_metric(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    _write_jsonl(
        path,
        [
            {"time": 1.0, "step": 1, "name": "reward_mean", "value": 0.5},
            {"time": 1.5, "step": 1, "name": "custom_metric", "value": 42},
            {"time": 2.0, "step": 2, "name": "reward_mean", "value": 0.75},
        ],
    )

    df = stream_jsonl_to_dataframe(path)

    assert "custom_metric" in df.columns
    assert df.loc[df["step"] == 1, "custom_metric"].iat[0] == pytest.approx(42)


def test_stream_jsonl_to_dataframe_accepts_integer_wall_time(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    _write_jsonl(
        path,
        [
            {"time": 1, "step": 1, "name": "reward_mean", "value": 0.5},
            {"time": 2, "step": 2, "name": "reward_mean", "value": 0.75},
        ],
    )

    df = stream_jsonl_to_dataframe(path)

    assert df["wall_time"].tolist() == pytest.approx([1.0, 2.0])


def test_stream_jsonl_to_dataframe_warns_on_invalid_json(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "metrics.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": 1.0, "step": 1, "name": "reward_mean", "value": 0.1}) + "\n")
        handle.write("not-json\n")
        handle.write(json.dumps({"time": 2.0, "step": 2, "name": "reward_mean", "value": 0.2}) + "\n")

    caplog.set_level(logging.WARNING)
    df = stream_jsonl_to_dataframe(path)

    assert len(df) == 2
    assert any("Skipped 1 invalid JSONL line" in message for message in caplog.messages)


def test_stream_jsonl_to_dataframe_with_preset(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    _write_jsonl(
        path,
        [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "global_step": 1,
                "metric": "reward_mean",
                "metric_value": 0.5,
            },
            {
                "timestamp": "2024-01-01T00:00:01Z",
                "global_step": 2,
                "metric": "reward_mean",
                "metric_value": 0.75,
            },
        ],
    )

    df = stream_jsonl_to_dataframe(path, preset="trl")

    assert list(df["step"]) == [1, 2]
    assert df["reward_mean"].tolist() == pytest.approx([0.5, 0.75])


def test_stream_jsonl_to_dataframe_missing_step(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    _write_jsonl(
        path,
        [
            {"time": 1.0, "name": "reward_mean", "value": 0.5},
        ],
    )

    with pytest.raises(ValidationError) as excinfo:
        stream_jsonl_to_dataframe(path)

    assert "Missing step" in str(excinfo.value)


def test_stream_jsonl_to_dataframe_empty_file(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text("")

    caplog.set_level(logging.INFO)
    df = stream_jsonl_to_dataframe(path)

    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert any("No events found" in message for message in caplog.messages)
