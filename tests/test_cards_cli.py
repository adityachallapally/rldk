"""Tests for the `rldk card` CLI command."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from rldk.cli import app


@pytest.fixture
def runner() -> CliRunner:
    """Provide a CLI runner for invoking Typer commands."""

    return CliRunner()


def test_reward_card_cli_from_jsonl(tmp_path: Path, runner: CliRunner) -> None:
    """The reward card command should normalize JSONL inputs and emit artifacts."""

    source = Path("tests/fixtures/phase_ab/stream_small.jsonl")
    output_dir = tmp_path / "cards"

    result = runner.invoke(
        app,
        [
            "card",
            "reward",
            str(source),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.stderr or result.stdout
    assert (output_dir / "reward_card.json").exists()
    assert (output_dir / "reward_card.png").exists()


def test_reward_card_cli_warns_when_reward_missing(
    tmp_path: Path, runner: CliRunner
) -> None:
    """A friendly warning should be emitted when reward metrics are absent."""

    source = tmp_path / "no_reward.jsonl"
    source.write_text(
        '{"time": 1700000001, "step": 1, "name": "kl_mean", "value": 0.1}\n'
    )

    output_dir = tmp_path / "cards"

    result = runner.invoke(
        app,
        [
            "card",
            "reward",
            str(source),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.stderr or result.stdout
    warning_output = (result.stderr or "") + (result.stdout or "")
    assert "Use --preset or --field-map" in warning_output
    assert (output_dir / "reward_card.json").exists()
    assert (output_dir / "reward_card.png").exists()

