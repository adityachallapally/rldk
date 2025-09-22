import json
from pathlib import Path

import pandas as pd
import pytest

from src.rldk.io.reward_writers import write_reward_health_summary
from src.rldk.io.consolidated_writers import write_reward_health_summary as consolidated_summary
from src.rldk.reward.health_analysis import RewardHealthReport
from src.rldk.reward.length_bias import LengthBiasMetrics


@pytest.fixture()
def sample_report() -> RewardHealthReport:
    metrics = LengthBiasMetrics(
        valid_sample_count=5,
        pearson_correlation=0.42,
        spearman_correlation=0.4,
        variance_explained=0.2,
        bias_severity=0.55,
        recommendations=["Review reward model prompts for length bias."],
    )
    return RewardHealthReport(
        passed=False,
        drift_detected=False,
        saturation_issues=["High zero clustering"],
        calibration_score=0.65,
        shortcut_signals=["Length bias detected (severity 0.55)"],
        label_leakage_risk=0.1,
        fixes=["Review reward model prompts for length bias."],
        drift_metrics=pd.DataFrame(),
        calibration_details={},
        shortcut_analysis={},
        saturation_analysis={},
        length_bias_detected=True,
        length_bias_metrics=metrics,
        length_bias_recommendations=metrics.recommendations,
    )


def _load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@pytest.mark.parametrize("writer_func", [write_reward_health_summary, consolidated_summary])
def test_reward_health_summary_serializes_length_bias(tmp_path: Path, sample_report: RewardHealthReport, writer_func) -> None:
    writer_func(sample_report, tmp_path)
    summary_path = tmp_path / "reward_health_summary.json"
    assert summary_path.exists()

    payload = _load_summary(summary_path)

    assert payload["length_bias_detected"] is True
    assert pytest.approx(payload["length_bias_metrics"]["bias_severity"], rel=1e-6) == 0.55
    assert payload["length_bias_recommendations"] == [
        "Review reward model prompts for length bias."
    ]
