import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from rldk.cli import app


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def _write_reward_jsonl(path: Path) -> None:
    records = [
        {"time": 1.0, "step": 1, "name": "reward", "value": 0.5},
        {"time": 1.0, "step": 1, "name": "kl", "value": 0.1},
        {"time": 2.0, "step": 2, "name": "reward", "value": 0.6},
        {"time": 2.0, "step": 2, "name": "kl", "value": 0.12},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_reward_health_cli_normalizes_jsonl(tmp_path: Path, runner: CliRunner) -> None:
    jsonl_path = tmp_path / "metrics.jsonl"
    _write_reward_jsonl(jsonl_path)

    output_dir = tmp_path / "outputs"
    mapping = json.dumps({"reward": "reward_mean", "kl": "kl_mean"})
    result = runner.invoke(
        app,
        [
            "reward-health",
            "--run",
            str(jsonl_path),
            "--output-dir",
            str(output_dir),
            "--field-map",
            mapping,
        ],
    )

    assert result.exit_code == 0, result.stdout

    summary_path = output_dir / "reward_health_summary.json"
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "calibration_score" in summary
    assert "overoptimization" in summary


def test_reward_health_cli_missing_reward(tmp_path: Path, runner: CliRunner) -> None:
    csv_path = tmp_path / "metrics.csv"
    pd.DataFrame({"step": [1, 2], "kl_mean": [0.1, 0.2]}).to_csv(csv_path, index=False)

    output_dir = tmp_path / "outputs"
    result = runner.invoke(
        app,
        [
            "reward-health",
            "--run",
            str(csv_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code != 0
    error_output = result.stderr or result.stdout
    assert "Use --preset or --field-map" in error_output


def test_reward_health_cli_directory_field_map(tmp_path: Path, runner: CliRunner) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pd.DataFrame(
        {
            "progress": [1, 2, 3],
            "reward_metric": [0.5, 0.55, 0.6],
            "kl_metric": [0.1, 0.11, 0.12],
        }
    ).to_csv(run_dir / "metrics.csv", index=False)

    output_dir = tmp_path / "outputs"
    mapping = json.dumps(
        {
            "progress": "step",
            "reward_metric": "reward_mean",
            "kl_metric": "kl_mean",
        }
    )

    result = runner.invoke(
        app,
        [
            "reward-health",
            "--run",
            str(run_dir),
            "--output-dir",
            str(output_dir),
            "--field-map",
            mapping,
        ],
    )

    assert result.exit_code == 0, result.stdout
