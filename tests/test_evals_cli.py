from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from rldk.cli import app


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def test_evals_cli_runs_quick_suite_from_training_metrics(
    tmp_path: Path, runner: CliRunner
) -> None:
    metrics = pd.DataFrame(
        {
            "step": [1, 2, 3, 4],
            "reward_mean": [0.2, 0.25, 0.3, 0.35],
            "kl_mean": [0.1, 0.11, 0.09, 0.1],
        }
    )
    metrics_path = tmp_path / "training_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    output_path = tmp_path / "results.json"
    result = runner.invoke(
        app,
        [
            "evals",
            "evaluate",
            str(metrics_path),
            "--suite",
            "quick",
            "--min-samples",
            "0",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert output_path.exists()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    evaluations = payload.get("evaluations")
    assert isinstance(evaluations, dict)
    assert evaluations, "expected per-metric status entries in evaluations output"
    assert payload.get("summary", {}).get("total_evaluations") == len(evaluations)
