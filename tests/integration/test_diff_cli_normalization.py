"""Integration tests for the diff CLI using normalized training metrics."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _cli_command(*args: str) -> list[str]:
    """Build a python -m rldk.cli command with the provided arguments."""

    return [sys.executable, "-m", "rldk.cli", *args]


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_diff_cli_jsonl(tmp_path: Path) -> None:
    """The diff CLI normalizes JSONL event streams and computes signal deltas."""

    run_a = tmp_path / "run_a.jsonl"
    run_b = tmp_path / "run_b.jsonl"

    events_a = []
    events_b = []
    for step in range(1, 4):
        events_a.append({"step": step, "name": "reward_mean", "value": 1.0 + step * 0.05})
        events_a.append({"step": step, "name": "kl_mean", "value": 0.01 * step})

        events_b.append({"step": step, "name": "reward_mean", "value": 1.1 + step * 0.05})
        events_b.append({"step": step, "name": "kl_mean", "value": 0.015 * step})

    _write_jsonl(run_a, events_a)
    _write_jsonl(run_b, events_b)

    output_dir = tmp_path / "jsonl_diff"
    result = subprocess.run(
        _cli_command(
            "diff",
            "--a",
            str(run_a),
            "--b",
            str(run_b),
            "--signals",
            "reward_mean",
            "--signals",
            "kl_mean",
            "--output-dir",
            str(output_dir),
        ),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    report_path = output_dir / "diff_report.json"
    assert report_path.exists(), result.stdout

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["summary"]["signals_compared"] == 2
    assert report["summary"]["verdict"] == "differs"

    reward_entry = next(item for item in report["signals"] if item["signal"] == "reward_mean")
    assert reward_entry["steps_compared"] == 3
    assert reward_entry["status"] == "differs"
    assert reward_entry["max_abs_delta"] > 0


def test_diff_cli_csv(tmp_path: Path) -> None:
    """The diff CLI accepts pre-normalized CSV tables."""

    run_a = tmp_path / "run_a.csv"
    run_b = tmp_path / "run_b.csv"

    df_a = pd.DataFrame(
        {
            "step": [1, 2, 3],
            "reward_mean": [1.0, 1.05, 1.1],
            "kl_mean": [0.01, 0.02, 0.03],
        }
    )
    df_b = df_a.copy()

    df_a.to_csv(run_a, index=False)
    df_b.to_csv(run_b, index=False)

    output_dir = tmp_path / "csv_diff"
    result = subprocess.run(
        _cli_command(
            "diff",
            "--a",
            str(run_a),
            "--b",
            str(run_b),
            "--signals",
            "reward_mean",
            "--signals",
            "kl_mean",
            "--output-dir",
            str(output_dir),
        ),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    report = json.loads((output_dir / "diff_report.json").read_text(encoding="utf-8"))
    assert report["summary"]["verdict"] == "match"
    assert report["summary"]["signals_with_differences"] == 0
    for entry in report["signals"]:
        assert entry["status"] in {"ok", "no_overlap"}
        assert entry["max_abs_delta"] in (0, None)


def test_diff_cli_missing_signal(tmp_path: Path) -> None:
    """Requesting a missing signal produces a helpful validation error."""

    run_a = tmp_path / "run_a.csv"
    run_b = tmp_path / "run_b.csv"

    df = pd.DataFrame({"step": [1, 2], "reward_mean": [1.0, 1.1]})
    df.to_csv(run_a, index=False)
    df.to_csv(run_b, index=False)

    result = subprocess.run(
        _cli_command(
            "diff",
            "--a",
            str(run_a),
            "--b",
            str(run_b),
            "--signals",
            "reward_mean",
            "--signals",
            "accuracy",
        ),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Signal 'accuracy' not found" in result.stderr
    assert "Available columns" in result.stderr
