from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from rldk.acceptance.summary import summarize_from_artifacts


def _write_reward_series(path: Path, *, count: int = 1001, start: float = 0.0, delta: float = 0.1) -> None:
    step = delta / (count - 1)
    with path.open("w") as handle:
        for idx in range(count):
            value = start + step * idx
            payload = {"name": "reward_mean", "value": value}
            handle.write(json.dumps(payload) + "\n")


def _copy_fixture_tree(target: Path, fixture_name: str) -> None:
    source = Path(__file__).parent.parent / "fixtures" / "fullscale" / fixture_name
    for file_path in source.rglob("*"):
        if file_path.is_dir():
            continue
        relative = file_path.relative_to(source)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(file_path, destination)


@pytest.mark.parametrize("fixture", ["healthy", "failing"])
def test_summarize_from_artifacts(tmp_path: Path, fixture: str) -> None:
    artifact_dir = tmp_path / fixture
    artifact_dir.mkdir()

    _write_reward_series(artifact_dir / "run.jsonl")
    _write_reward_series(artifact_dir / "baseline.jsonl")

    _copy_fixture_tree(artifact_dir, fixture)

    result = summarize_from_artifacts(artifact_dir)

    if fixture == "healthy":
        assert result.ok, "Expected healthy fixture to pass gating"
        assert result.lines[-1] == "PASS"
    else:
        assert not result.ok, "Expected failing fixture to trip acceptance gate"
        summary_text = "\n".join(result.lines)
        assert "Alert count" in summary_text
        assert "Reward card gate failed" in summary_text
