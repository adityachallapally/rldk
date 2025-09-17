"""Integration checks for reward CLI behavior in Phase A/B."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from rldk.testing.cli_detect import (
    detect_reward_drift_cmd,
    detect_reward_health_cmd,
)


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "phase_ab"


def _find_numeric_field(payload: dict, candidates: Sequence[str]) -> float:
    for key in candidates:
        if key in payload and isinstance(payload[key], (int, float)):
            value = float(payload[key])
            if np.isfinite(value):
                return value
    pytest.fail(
        "No numeric field found in payload; checked keys: "
        + ", ".join(candidates)
        + f" in {sorted(payload)}"
    )


def _run_command(command: Sequence[str], cwd: Path, **kwargs) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        **kwargs,
    )
    return result


@pytest.fixture(scope="session")
def reward_health_cmd() -> Sequence[str]:
    return detect_reward_health_cmd()


@pytest.fixture(scope="session")
def reward_drift_cmd() -> Sequence[str]:
    return detect_reward_drift_cmd()


@pytest.mark.parametrize("source_name", ["stream_small.jsonl", "table_small.csv"])
def test_reward_health_cli_normalizes_inputs(
    tmp_path: Path, reward_health_cmd: Sequence[str], source_name: str
) -> None:
    source_path = FIXTURES / source_name
    out_dir = tmp_path / f"out_{source_path.stem}"

    command = [*reward_health_cmd, "--run", str(source_path), "--output-dir", str(out_dir)]
    result = _run_command(command, cwd=tmp_path)

    assert result.returncode == 0, result.stderr or result.stdout
    summary_path = out_dir / "reward_health_summary.json"
    assert summary_path.exists(), summary_path

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    _find_numeric_field(
        payload,
        [
            "health",
            "health_score",
            "score",
            "overall",
            "overall_score",
            "calibration_score",
            "label_leakage_risk",
        ],
    )


def test_reward_health_cli_missing_reward(tmp_path: Path, reward_health_cmd: Sequence[str]) -> None:
    source_path = tmp_path / "no_reward.jsonl"
    records = [
        {"time": 1.0, "step": 1, "name": "kl_mean", "value": 0.1},
        {"time": 1.5, "step": 1, "name": "loss", "value": 0.2},
    ]
    with source_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    out_dir = tmp_path / "no_reward_out"
    command = [*reward_health_cmd, "--run", str(source_path), "--output-dir", str(out_dir)]
    result = _run_command(command, cwd=tmp_path)

    assert result.returncode != 0
    stderr = result.stderr.lower() or result.stdout.lower()
    assert "reward" in stderr
    assert "preset" in stderr or "field-map" in stderr or "field map" in stderr


def test_reward_drift_cli_score_mode(tmp_path: Path, reward_drift_cmd: Sequence[str]) -> None:
    scores_a = FIXTURES / "scores_a.jsonl"
    scores_b = FIXTURES / "scores_b.jsonl"
    command = [
        *reward_drift_cmd,
        "--scores-a",
        str(scores_a),
        "--scores-b",
        str(scores_b),
    ]
    result = _run_command(command, cwd=tmp_path)

    assert result.returncode == 0, result.stderr or result.stdout

    report_path = tmp_path / "rldk_reports" / "reward_drift.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    _find_numeric_field(payload, ["drift", "drift_magnitude", "effect_size"])
