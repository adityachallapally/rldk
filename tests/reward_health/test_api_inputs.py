import json
from pathlib import Path

import pandas as pd
import pytest

from rldk import HealthAnalysisResult, reward_health


@pytest.fixture()
def base_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "step": [1, 2, 3],
            "reward_mean": [0.5, 0.55, 0.6],
            "kl_mean": [0.1, 0.11, 0.09],
        }
    )


@pytest.fixture()
def dataframe_result(base_metrics: pd.DataFrame) -> HealthAnalysisResult:
    return reward_health(base_metrics, reward_col="reward_mean", step_col="step")


def test_reward_health_accepts_dataframe(dataframe_result: HealthAnalysisResult) -> None:
    assert isinstance(dataframe_result, HealthAnalysisResult)
    assert list(dataframe_result.metrics["step"]) == [1, 2, 3]


def test_reward_health_accepts_list(base_metrics: pd.DataFrame, dataframe_result: HealthAnalysisResult) -> None:
    list_result = reward_health(
        base_metrics.to_dict(orient="records"),
        reward_col="reward_mean",
        step_col="step",
    )
    columns = ["step", "reward_mean", "kl_mean"]
    pd.testing.assert_frame_equal(
        list_result.metrics[columns], dataframe_result.metrics[columns]
    )
    assert list_result.report.passed == dataframe_result.report.passed
    assert list_result.report.drift_detected == dataframe_result.report.drift_detected
    assert list_result.report.saturation_issues == dataframe_result.report.saturation_issues
    assert list_result.report.calibration_score == pytest.approx(
        dataframe_result.report.calibration_score
    )
    assert list_result.report.shortcut_signals == dataframe_result.report.shortcut_signals
    assert list_result.report.label_leakage_risk == pytest.approx(
        dataframe_result.report.label_leakage_risk
    )
    assert list_result.report.fixes == dataframe_result.report.fixes
    assert list_result.report.saturation_analysis == dataframe_result.report.saturation_analysis
    assert list_result.report.shortcut_analysis == dataframe_result.report.shortcut_analysis
    assert list_result.report.calibration_details == dataframe_result.report.calibration_details
    assert (
        list_result.report.length_bias_detected
        == dataframe_result.report.length_bias_detected
    )
    assert (
        list_result.report.length_bias_recommendations
        == dataframe_result.report.length_bias_recommendations
    )
    assert list_result.report.length_bias_metrics.to_dict() == dataframe_result.report.length_bias_metrics.to_dict()
    assert list_result.reference_metrics is None
    if list_result.report.drift_metrics is None:
        assert dataframe_result.report.drift_metrics is None
    else:
        pd.testing.assert_frame_equal(
            list_result.report.drift_metrics, dataframe_result.report.drift_metrics
        )


def test_reward_health_accepts_jsonl(tmp_path: Path, base_metrics: pd.DataFrame, dataframe_result: HealthAnalysisResult) -> None:
    jsonl_path = tmp_path / "metrics.jsonl"
    records = []
    for _, row in base_metrics.iterrows():
        step = int(row["step"])
        records.append(
            {"time": float(step), "step": step, "name": "reward", "value": float(row["reward_mean"])}
        )
        records.append(
            {"time": float(step), "step": step, "name": "kl", "value": float(row["kl_mean"])}
        )

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    jsonl_result = reward_health(
        jsonl_path,
        reward_col="reward_mean",
        step_col="step",
        field_map={"reward": "reward_mean", "kl": "kl_mean"},
    )
    columns = ["step", "reward_mean", "kl_mean"]
    pd.testing.assert_frame_equal(
        jsonl_result.metrics[columns], dataframe_result.metrics[columns]
    )
    assert jsonl_result.report.passed == dataframe_result.report.passed
    assert jsonl_result.report.drift_detected == dataframe_result.report.drift_detected
    assert jsonl_result.report.saturation_issues == dataframe_result.report.saturation_issues
    assert jsonl_result.report.calibration_score == pytest.approx(
        dataframe_result.report.calibration_score
    )
    assert jsonl_result.report.shortcut_signals == dataframe_result.report.shortcut_signals
    assert jsonl_result.report.label_leakage_risk == pytest.approx(
        dataframe_result.report.label_leakage_risk
    )
    assert jsonl_result.report.fixes == dataframe_result.report.fixes
    assert jsonl_result.report.saturation_analysis == dataframe_result.report.saturation_analysis
    assert jsonl_result.report.shortcut_analysis == dataframe_result.report.shortcut_analysis
    assert jsonl_result.report.calibration_details == dataframe_result.report.calibration_details
    assert (
        jsonl_result.report.length_bias_detected
        == dataframe_result.report.length_bias_detected
    )
    assert (
        jsonl_result.report.length_bias_recommendations
        == dataframe_result.report.length_bias_recommendations
    )
    assert jsonl_result.report.length_bias_metrics.to_dict() == dataframe_result.report.length_bias_metrics.to_dict()
    assert jsonl_result.reference_metrics is None
    if jsonl_result.report.drift_metrics is None:
        assert dataframe_result.report.drift_metrics is None
    else:
        pd.testing.assert_frame_equal(
            jsonl_result.report.drift_metrics, dataframe_result.report.drift_metrics
        )


def test_health_analysis_to_dict_includes_length_bias(
    dataframe_result: HealthAnalysisResult,
) -> None:
    payload = dataframe_result.to_dict()
    report = payload["report"]

    assert "length_bias_detected" in report
    assert "length_bias_metrics" in report
    assert "length_bias_recommendations" in report
    assert isinstance(report["length_bias_metrics"], dict)
