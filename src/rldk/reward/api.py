"""User-facing reward health helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import pandas as pd

from ..ingest.training_metrics_normalizer import normalize_training_metrics
from ..utils.error_handling import ValidationError
from .health_analysis import OveroptimizationAnalysis, RewardHealthReport, health
from .length_bias import LengthBiasMetrics

TrainingMetricsInput = Union[pd.DataFrame, Sequence[Mapping[str, Any]], str, Path]


@dataclass
class HealthAnalysisResult:
    """Container for reward health analysis results and normalized metrics."""

    report: RewardHealthReport
    metrics: pd.DataFrame
    reference_metrics: Optional[pd.DataFrame] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation of the analysis."""

        report_dict: Dict[str, Any] = {
            "passed": self.report.passed,
            "drift_detected": self.report.drift_detected,
            "saturation_issues": list(self.report.saturation_issues),
            "calibration_score": self.report.calibration_score,
            "shortcut_signals": list(self.report.shortcut_signals),
            "label_leakage_risk": self.report.label_leakage_risk,
            "fixes": list(self.report.fixes),
            "saturation_analysis": self.report.saturation_analysis,
            "shortcut_analysis": self.report.shortcut_analysis,
            "calibration_details": self.report.calibration_details,
            "length_bias_detected": self.report.length_bias_detected,
            "length_bias_recommendations": list(
                self.report.length_bias_recommendations
            ),
        }

        metrics_value = self.report.length_bias_metrics
        metrics_dict: Dict[str, Any] = {}
        if isinstance(metrics_value, LengthBiasMetrics):
            metrics_dict = metrics_value.to_dict()
        elif isinstance(metrics_value, dict):
            metrics_dict = metrics_value
        elif metrics_value is not None and hasattr(metrics_value, "to_dict"):
            metrics_dict = metrics_value.to_dict()  # type: ignore[assignment]
        report_dict["length_bias_metrics"] = metrics_dict

        if (
            self.report.drift_metrics is not None
            and isinstance(self.report.drift_metrics, pd.DataFrame)
            and not self.report.drift_metrics.empty
        ):
            report_dict["drift_metrics"] = self.report.drift_metrics.to_dict(orient="records")
        else:
            report_dict["drift_metrics"] = []

        overopt = getattr(self.report, "overoptimization", None)
        if isinstance(overopt, OveroptimizationAnalysis):
            report_dict["overoptimization"] = overopt.to_dict()
        elif hasattr(overopt, "to_dict"):
            report_dict["overoptimization"] = overopt.to_dict()
        else:
            report_dict["overoptimization"] = {}

        return {
            "report": report_dict,
            "metrics": self.metrics.to_dict(orient="records"),
            "reference_metrics": (
                self.reference_metrics.to_dict(orient="records")
                if isinstance(self.reference_metrics, pd.DataFrame)
                else None
            ),
        }


def _ensure_column_present(df: pd.DataFrame, column: str, kind: str) -> None:
    if column not in df.columns:
        raise ValidationError(
            f"{kind.capitalize()} column '{column}' not found after normalization",
            suggestion=(
                "Provide a field_map that points your source column to the canonical name, "
                "for example {'reward': 'reward_mean'}"
            ),
            error_code=f"MISSING_{kind.upper()}_COLUMN",
        )
    if df[column].dropna().empty:
        raise ValidationError(
            f"Normalized {kind} column '{column}' is empty",
            suggestion="Ensure the source data includes non-null values for this column",
            error_code=f"EMPTY_{kind.upper()}_COLUMN",
        )


def reward_health(
    run_data: TrainingMetricsInput,
    reference_data: Optional[TrainingMetricsInput] = None,
    *,
    field_map: Optional[Dict[str, str]] = None,
    reward_col: str = "reward_mean",
    step_col: str = "step",
    threshold_drift: float = 0.1,
    threshold_saturation: float = 0.8,
    threshold_calibration: float = 0.7,
    threshold_shortcut: float = 0.6,
    threshold_leakage: float = 0.3,
    response_col: Optional[str] = None,
    length_col: Optional[str] = None,
    threshold_length_bias: float = 0.4,
    enable_length_bias_detection: bool = True,
    gold_metrics: Optional[TrainingMetricsInput] = None,
    gold_metric_col: Optional[str] = None,
    overoptimization_window: int = 100,
    overoptimization_delta_threshold: float = 0.2,
    overoptimization_min_samples: int = 100,
) -> HealthAnalysisResult:
    """Run reward health analysis with flexible input formats."""

    run_metrics = normalize_training_metrics(run_data, field_map=field_map)

    if run_metrics.empty:
        raise ValidationError(
            "Normalized run data is empty",
            suggestion="Ensure the source contains reward metrics",
            error_code="EMPTY_RUN_DATA",
        )

    _ensure_column_present(run_metrics, step_col, "step")
    _ensure_column_present(run_metrics, reward_col, "reward")

    reference_metrics: Optional[pd.DataFrame] = None
    if reference_data is not None:
        reference_metrics = normalize_training_metrics(reference_data, field_map=field_map)
        if reference_metrics.empty:
            raise ValidationError(
                "Normalized reference data is empty",
                suggestion="Ensure the reference source contains reward metrics",
                error_code="EMPTY_REFERENCE_DATA",
            )
        _ensure_column_present(reference_metrics, step_col, "step")
        _ensure_column_present(reference_metrics, reward_col, "reward")

    gold_metrics_df: Optional[pd.DataFrame] = None
    if gold_metrics is not None:
        gold_metrics_df = normalize_training_metrics(gold_metrics, field_map=field_map)

    report = health(
        run_data=run_metrics,
        reference_data=reference_metrics,
        reward_col=reward_col,
        step_col=step_col,
        threshold_drift=threshold_drift,
        threshold_saturation=threshold_saturation,
        threshold_calibration=threshold_calibration,
        threshold_shortcut=threshold_shortcut,
        threshold_leakage=threshold_leakage,
        response_col=response_col,
        length_col=length_col,
        threshold_length_bias=threshold_length_bias,
        enable_length_bias_detection=enable_length_bias_detection,
        gold_metrics=gold_metrics_df,
        gold_metric_col=gold_metric_col,
        overoptimization_window=overoptimization_window,
        overoptimization_delta_threshold=overoptimization_delta_threshold,
        overoptimization_min_samples=overoptimization_min_samples,
    )

    return HealthAnalysisResult(report=report, metrics=run_metrics, reference_metrics=reference_metrics)


__all__ = ["HealthAnalysisResult", "reward_health"]
