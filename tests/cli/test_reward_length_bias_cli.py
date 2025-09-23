"""CLI smoke test for the reward length bias command."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from rldk.cli import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_reward_length_bias_cli(tmp_path: Path, runner: CliRunner) -> None:
    data = pd.DataFrame(
        {
            "step": range(1, 21),
            "run_id": ["cli-length" for _ in range(1, 21)],
            "reward_mean": [0.2 * idx for idx in range(1, 21)],
            "tokens_out": [idx * 5 for idx in range(1, 21)],
            "response_text": ["x" * (idx * 5) for idx in range(1, 21)],
        }
    )
    run_path = tmp_path / "metrics.jsonl"
    with run_path.open("w", encoding="utf-8") as fp:
        for record in data.to_dict(orient="records"):
            fp.write(json.dumps(record) + "\n")

    output_dir = tmp_path / "reports"

    result = runner.invoke(
        app,
        [
            "reward",
            "length-bias",
            "--run-path",
            str(run_path),
            "--adapter",
            "flexible",
            "--response-col",
            "response_text",
            "--reward-col",
            "reward_mean",
            "--length-col",
            "tokens_out",
            "--threshold",
            "0.25",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.stdout

    assert "Length bias evaluation summary:" in result.stdout
    assert "Report saved to:" in result.stdout

    json_section = result.stdout.split("JSON report:\n", 1)[1]
    json_payload = json_section.strip().split("\n\n", 1)[0]
    parsed = json.loads(json_payload)

    assert parsed["metrics"]["bias_severity"] is not None

    report_path = output_dir / "length_bias_report.json"
    assert report_path.exists()
    saved = json.loads(report_path.read_text())
    assert saved["metrics"]["bias_severity"] == parsed["metrics"]["bias_severity"]


def test_reward_length_bias_cli_generates_card(tmp_path: Path, runner: CliRunner) -> None:
    data = pd.DataFrame(
        {
            "step": range(1, 16),
            "run_id": ["cli-card"] * 15,
            "reward_mean": [0.1 * idx for idx in range(1, 16)],
            "tokens_out": [idx * 4 for idx in range(1, 16)],
            "response_text": ["y" * (idx * 4) for idx in range(1, 16)],
        }
    )
    run_path = tmp_path / "metrics.jsonl"
    with run_path.open("w", encoding="utf-8") as fp:
        for record in data.to_dict(orient="records"):
            fp.write(json.dumps(record) + "\n")

    output_dir = tmp_path / "reports"

    result = runner.invoke(
        app,
        [
            "reward",
            "length-bias",
            "--run-path",
            str(run_path),
            "--response-col",
            "response_text",
            "--reward-col",
            "reward_mean",
            "--length-col",
            "tokens_out",
            "--generate-card",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.stdout

    card_dir = Path("runs/cli-card/rldk_cards/length_bias")
    try:
        assert (card_dir / "length_bias_card.json").exists()
        assert (card_dir / "length_bias_card.png").exists()

        copied_json = output_dir / "length_bias_card.json"
        copied_png = output_dir / "length_bias_card.png"
        assert copied_json.exists()
        assert copied_png.exists()
    finally:
        if card_dir.exists():
            shutil.rmtree(card_dir.parent.parent, ignore_errors=True)
