"""Catastrophic forgetting regression evaluation metric."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ...config import EvaluationConfig, get_eval_config

logger = logging.getLogger(__name__)

# Column name candidates used when explicit configuration is not provided.
TASK_COLUMN_CANDIDATES = (
    "task",
    "task_id",
    "benchmark",
    "benchmark_id",
    "dataset",
    "evaluation_name",
)

SCORE_COLUMN_CANDIDATES = (
    "score",
    "reward",
    "reward_mean",
    "metric",
    "metric_value",
    "accuracy",
    "f1",
)


@dataclass
class BaselineSummary:
    """Container for baseline statistics used during regression checks."""

    mean: float
    std: float
    count: int
    timestamp: Optional[str] = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BaselineSummary":
        """Create a :class:`BaselineSummary` from a generic mapping."""

        mean = payload.get("mean")
        std = payload.get("std") or payload.get("stdev") or payload.get("stddev") or 0.0
        count = payload.get("count") or payload.get("sample_count") or payload.get("n") or 0
        timestamp = payload.get("timestamp") or payload.get("evaluated_at")

        if mean is None:
            raise ValueError("Baseline summary must include a mean score")

        return cls(mean=float(mean), std=float(std), count=int(count), timestamp=timestamp)


def _resolve_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    """Return the first candidate that exists in the provided column names."""

    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _prepare_baselines(
    baseline_data: Any,
    task_column: Optional[str],
    overrides: Dict[str, Any],
) -> Dict[str, BaselineSummary]:
    """Normalize baseline inputs into a mapping of task -> :class:`BaselineSummary`."""

    if baseline_data is None:
        return {}

    if isinstance(baseline_data, Mapping):
        baselines: Dict[str, BaselineSummary] = {}
        for task, payload in baseline_data.items():
            try:
                baselines[str(task)] = BaselineSummary.from_mapping(payload)
            except (TypeError, ValueError) as exc:
                logger.warning("Skipping malformed baseline summary for task '%s': %s", task, exc)
        return baselines

    if isinstance(baseline_data, pd.DataFrame):
        baseline_task_column = overrides.get("baseline_task_column") or task_column
        if baseline_task_column is None:
            baseline_task_column = _resolve_column(
                baseline_data.columns, ("task", "task_id", "benchmark", "benchmark_id")
            )

        if baseline_task_column is None or baseline_task_column not in baseline_data.columns:
            logger.warning(
                "Baseline DataFrame does not include a recognizable task column; skipping baselines"
            )
            return {}

        mean_column = overrides.get("baseline_mean_column") or _resolve_column(
            baseline_data.columns, ("mean", "score_mean", "avg", "average")
        )
        std_column = overrides.get("baseline_std_column") or _resolve_column(
            baseline_data.columns, ("std", "stdev", "stddev")
        )
        count_column = overrides.get("baseline_count_column") or _resolve_column(
            baseline_data.columns, ("count", "sample_count", "n")
        )
        timestamp_column = overrides.get("baseline_timestamp_column") or _resolve_column(
            baseline_data.columns, ("timestamp", "evaluated_at", "updated_at")
        )

        baselines: Dict[str, BaselineSummary] = {}
        for _, row in baseline_data.iterrows():
            task_identifier = str(row[baseline_task_column])
            payload = {
                "mean": row.get(mean_column, np.nan) if mean_column else np.nan,
                "std": row.get(std_column, 0.0) if std_column else 0.0,
                "count": row.get(count_column, 0) if count_column else 0,
            }
            if timestamp_column:
                payload["timestamp"] = row.get(timestamp_column)
            try:
                baselines[task_identifier] = BaselineSummary.from_mapping(payload)
            except (TypeError, ValueError) as exc:
                logger.warning("Skipping malformed baseline summary for task '%s': %s", task_identifier, exc)
        return baselines

    logger.warning("Unsupported baseline data type '%s'; expected mapping or DataFrame", type(baseline_data))
    return {}


def _confidence_interval(values: np.ndarray, confidence_level: float = 0.95) -> Tuple[float, float]:
    """Compute a confidence interval for the sample mean."""

    if values.size == 0:
        return (np.nan, np.nan)

    if values.size == 1:
        value = float(values[0])
        return (value, value)

    mean = float(values.mean())
    std = float(values.std(ddof=1))
    if std == 0 or np.isnan(std):
        return (mean, mean)

    se = std / np.sqrt(values.size)
    try:
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
    except Exception:  # pragma: no cover - scipy should always return a value
        z_score = 1.96
    margin = z_score * se
    return (mean - margin, mean + margin)


def evaluate_catastrophic_forgetting(
    data: pd.DataFrame,
    *,
    baseline_summaries: Optional[Any] = None,
    config: Optional[EvaluationConfig] = None,
    task_column: Optional[str] = None,
    score_column: Optional[str] = None,
    regression_threshold: Optional[float] = None,
    regression_z_threshold: Optional[float] = None,
    min_samples: Optional[int] = None,
    weighting_strategy: Optional[str] = None,
    task_weights: Optional[Mapping[str, float]] = None,
    **baseline_overrides: Any,
) -> Dict[str, Any]:
    """Evaluate catastrophic forgetting across benchmarks and tasks."""

    warnings: List[str] = []

    if data is None or data.empty:
        warnings.append("No evaluation data provided for catastrophic forgetting analysis")
        logger.warning(warnings[-1])
        return {
            "score": np.nan,
            "regressed_tasks": [],
            "stable_tasks": [],
            "missing_baselines": [],
            "recommendations": ["Provide evaluation runs with task and score columns to assess regression."],
            "warnings": warnings,
            "metadata": {
                "evaluated_tasks": 0,
                "used_samples": 0,
                "weighting_strategy": weighting_strategy or "baseline_count",
                "baseline_available": False,
            },
        }

    config = config or get_eval_config()
    regression_threshold = (
        regression_threshold if regression_threshold is not None else config.CATASTROPHIC_REGRESSION_THRESHOLD
    )
    regression_z_threshold = (
        regression_z_threshold
        if regression_z_threshold is not None
        else config.CATASTROPHIC_REGRESSION_Z_THRESHOLD
    )
    min_samples = min_samples if min_samples is not None else config.CATASTROPHIC_MIN_SAMPLES
    weighting_strategy = weighting_strategy or config.CATASTROPHIC_WEIGHTING_STRATEGY

    task_column = task_column or _resolve_column(data.columns, TASK_COLUMN_CANDIDATES)
    score_column = score_column or _resolve_column(data.columns, SCORE_COLUMN_CANDIDATES)

    if task_column is None:
        message = "Task or benchmark identifier column not found; expected one of task/task_id/benchmark"
        warnings.append(message)
        logger.warning(message)
    if score_column is None:
        message = "Score column not found; expected one of score/reward/metric"
        warnings.append(message)
        logger.warning(message)

    if task_column is None or score_column is None:
        return {
            "score": np.nan,
            "regressed_tasks": [],
            "stable_tasks": [],
            "missing_baselines": [],
            "recommendations": ["Ensure task identifiers and numeric scores are included in evaluation exports."],
            "warnings": warnings,
            "metadata": {
                "evaluated_tasks": 0,
                "used_samples": 0,
                "weighting_strategy": weighting_strategy,
                "baseline_available": bool(baseline_summaries),
            },
        }

    working = data.copy()
    working = working.dropna(subset=[task_column, score_column])
    working[score_column] = pd.to_numeric(working[score_column], errors="coerce")
    working = working.dropna(subset=[score_column])

    if working.empty:
        warnings.append("No numeric scores available after cleaning catastrophic forgetting inputs")
        logger.warning(warnings[-1])
        return {
            "score": np.nan,
            "regressed_tasks": [],
            "stable_tasks": [],
            "missing_baselines": [],
            "recommendations": ["Verify that evaluation exports include numeric scores for each task."],
            "warnings": warnings,
            "metadata": {
                "evaluated_tasks": 0,
                "used_samples": 0,
                "weighting_strategy": weighting_strategy,
                "baseline_available": bool(baseline_summaries),
            },
        }

    baselines = _prepare_baselines(baseline_summaries, task_column, baseline_overrides)
    if not baselines:
        warnings.append("No baseline summaries provided; catastrophic regression scoring may be incomplete")
        logger.warning(warnings[-1])

    regressed_tasks: List[Dict[str, Any]] = []
    stable_tasks: List[Dict[str, Any]] = []
    missing_baselines: List[str] = []
    weighted_scores: List[Tuple[float, float]] = []  # (weight, normalized_score)

    unique_tasks = working[task_column].astype(str).unique()
    evaluated_task_count = 0
    total_samples = 0

    for task_name, group in working.groupby(task_column):
        task_id = str(task_name)
        scores = group[score_column].to_numpy(dtype=float)
        sample_count = int(scores.size)
        total_samples += sample_count

        baseline = baselines.get(task_id)
        if baseline is None:
            missing_baselines.append(task_id)
            logger.info("No baseline available for task '%s'; excluding from regression score", task_id)
            continue

        evaluated_task_count += 1

        current_mean = float(np.mean(scores)) if scores.size else np.nan
        current_min = float(np.min(scores)) if scores.size else np.nan
        current_max = float(np.max(scores)) if scores.size else np.nan

        delta_mean = current_mean - baseline.mean
        delta_min = current_min - baseline.mean
        delta_max = current_max - baseline.mean

        if baseline.std and not np.isnan(baseline.std):
            z_score = delta_mean / max(baseline.std, 1e-8)
        else:
            z_score = np.nan

        ci_lower, ci_upper = _confidence_interval(scores)

        insufficient_samples = sample_count < min_samples
        if insufficient_samples:
            warning = (
                f"Task '{task_id}' has only {sample_count} samples (<{min_samples}); diagnostics reported without scoring"
            )
            warnings.append(warning)
            logger.info(warning)

        regression_flag = (
            (delta_mean <= regression_threshold)
            or (not np.isnan(z_score) and z_score <= regression_z_threshold)
        )

        task_payload = {
            "task": task_id,
            "current_mean": current_mean,
            "current_min": current_min,
            "current_max": current_max,
            "baseline_mean": baseline.mean,
            "baseline_std": baseline.std,
            "baseline_count": baseline.count,
            "delta_mean": delta_mean,
            "delta_min": delta_min,
            "delta_max": delta_max,
            "z_score": z_score,
            "sample_count": sample_count,
            "confidence_interval": {"lower": ci_lower, "upper": ci_upper},
            "timestamp": baseline.timestamp,
            "regression_threshold": regression_threshold,
            "regression_z_threshold": regression_z_threshold,
        }

        if regression_flag and not insufficient_samples:
            regressed_tasks.append(task_payload)
        else:
            task_payload["status"] = "insufficient_samples" if insufficient_samples else "stable"
            stable_tasks.append(task_payload)

        if insufficient_samples:
            continue

        if weighting_strategy == "sample_count":
            weight = float(sample_count)
        elif weighting_strategy == "baseline_count":
            weight = float(baseline.count or sample_count)
        elif weighting_strategy == "custom" and task_weights:
            weight = float(task_weights.get(task_id, sample_count))
        else:
            weight = 1.0

        if weight <= 0:
            logger.info("Ignoring non-positive weight for task '%s'", task_id)
            continue

        if regression_flag:
            normalized_score = 0.0
        elif delta_mean >= 0:
            normalized_score = 1.0
        else:
            normalized_score = max(0.0, 1.0 + (delta_mean / abs(regression_threshold)))

        weighted_scores.append((weight, normalized_score))

    if weighted_scores:
        weights, task_scores = zip(*weighted_scores)
        total_weight = float(np.sum(weights))
        if total_weight > 0:
            aggregate_score = float(np.sum(np.array(weights) * np.array(task_scores)) / total_weight)
        else:
            aggregate_score = np.nan
    else:
        aggregate_score = np.nan

    if np.isnan(aggregate_score):
        recommendations = [
            "Review baseline coverage and provide sufficient samples per task to compute regression scores.",
        ]
    elif regressed_tasks:
        top_regressions = sorted(regressed_tasks, key=lambda x: x["delta_mean"])[:3]
        regression_tasks = ", ".join(task["task"] for task in top_regressions)
        recommendations = [
            f"Re-run fine-tuning or rehearsal on tasks {regression_tasks} to mitigate catastrophic forgetting.",
            "Consider adding rehearsal samples or elastic weight consolidation for the affected domains.",
        ]
    else:
        recommendations = ["Maintain current rehearsal cadence; no catastrophic forgetting detected."]

    metadata = {
        "evaluated_tasks": evaluated_task_count,
        "observed_tasks": int(len(unique_tasks)),
        "used_samples": total_samples,
        "weighting_strategy": weighting_strategy,
        "regression_threshold": regression_threshold,
        "regression_z_threshold": regression_z_threshold,
        "min_samples": min_samples,
        "baseline_available": bool(baselines),
    }

    if missing_baselines:
        recommendations.append(
            "Collect or regenerate baseline summaries for tasks: " + ", ".join(sorted(missing_baselines))
        )

    return {
        "score": aggregate_score,
        "regressed_tasks": regressed_tasks,
        "stable_tasks": stable_tasks,
        "missing_baselines": sorted(missing_baselines),
        "recommendations": recommendations,
        "warnings": warnings,
        "metadata": metadata,
    }


__all__ = ["evaluate_catastrophic_forgetting"]

