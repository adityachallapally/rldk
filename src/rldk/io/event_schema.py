"""Normalized event schema for RL training runs."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from ..utils.error_handling import ValidationError

logger = logging.getLogger(__name__)

_TRAINING_METRIC_COLUMNS = [
    "step",
    "phase",
    "reward_mean",
    "reward_std",
    "kl_mean",
    "entropy_mean",
    "clip_frac",
    "grad_norm",
    "lr",
    "loss",
    "tokens_in",
    "tokens_out",
    "wall_time",
    "seed",
    "run_id",
]

_CORE_METRIC_FIELDS = [
    "reward_mean",
    "reward_std",
    "kl_mean",
    "entropy_mean",
    "clip_frac",
    "grad_norm",
    "lr",
    "loss",
]

_NETWORK_METRIC_FIELDS = [
    "network_bandwidth",
    "network_latency",
    "bandwidth_mbps",
    "latency_ms",
    "bandwidth_upload_mbps",
    "bandwidth_download_mbps",
    "total_bandwidth_mbps",
    "allreduce_bandwidth",
    "broadcast_bandwidth",
    "gather_bandwidth",
    "scatter_bandwidth",
    "packet_loss_percent",
    "network_errors",
    "dns_resolution_ms",
]

_NUMERIC_COLUMNS = {
    "reward_mean",
    "reward_std",
    "kl_mean",
    "entropy_mean",
    "clip_frac",
    "grad_norm",
    "lr",
    "loss",
    "tokens_in",
    "tokens_out",
    "wall_time",
}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return True
    try:
        return bool(pd.isna(value))  # type: ignore[arg-type]
    except Exception:
        return False


def _coerce_numeric(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        value_float = float(value)
        return value_float if math.isfinite(value_float) else None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            value_float = float(value)
        except ValueError:
            return None
        return value_float if math.isfinite(value_float) else None
    return None


def _coerce_wall_time(value: Any) -> Optional[float]:
    numeric_value = _coerce_numeric(value)
    if numeric_value is not None:
        return numeric_value
    if hasattr(value, "timestamp"):
        try:
            return float(value.timestamp())
        except (TypeError, ValueError):
            return None
    return None


def _stable_column_order(columns: Iterable[str]) -> List[str]:
    ordered = [column for column in _TRAINING_METRIC_COLUMNS if column in columns]
    extras = sorted(column for column in columns if column not in ordered)
    return ordered + extras


@dataclass
class Event:
    """Normalized event object representing a single training step."""

    step: int
    wall_time: float
    metrics: Dict[str, float]
    rng: Dict[str, Any]
    data_slice: Dict[str, Any]
    model_info: Dict[str, Any]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for JSON serialization."""
        return {
            "step": self.step,
            "wall_time": self.wall_time,
            "metrics": self.metrics,
            "rng": self.rng,
            "data_slice": self.data_slice,
            "model_info": self.model_info,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create event from dictionary."""
        return cls(
            step=data["step"],
            wall_time=data["wall_time"],
            metrics=data["metrics"],
            rng=data["rng"],
            data_slice=data["data_slice"],
            model_info=data["model_info"],
            notes=data.get("notes", []),
        )

    def to_json(self) -> str:
        """Serialize event to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "Event":
        """Create event from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


def create_event_from_row(
    row: Dict[str, Any], run_id: str, git_sha: Optional[str] = None
) -> Event:
    """Create an Event object from a training data row."""

    metrics: Dict[str, float] = {}
    for field in [*_CORE_METRIC_FIELDS, *_NETWORK_METRIC_FIELDS]:
        if field not in row:
            continue

        value = row[field]
        if _is_missing(value):
            continue

        numeric_value = _coerce_numeric(value)
        if numeric_value is not None:
            metrics[field] = numeric_value

    rng = {
        "seed": row.get("seed"),
        "python_hash_seed": row.get("python_hash_seed"),
        "torch_seed": row.get("torch_seed"),
        "numpy_seed": row.get("numpy_seed"),
        "random_seed": row.get("random_seed"),
    }

    data_slice = {
        "tokens_in": row.get("tokens_in"),
        "tokens_out": row.get("tokens_out"),
        "batch_size": row.get("batch_size"),
        "sequence_length": row.get("sequence_length"),
    }

    model_info = {
        "run_id": run_id,
        "git_sha": git_sha,
        "phase": row.get("phase") or "train",
        "model_name": row.get("model_name"),
        "model_size": row.get("model_size"),
        "optimizer": row.get("optimizer"),
        "scheduler": row.get("scheduler"),
    }

    notes: List[str] = []
    clip_frac = _coerce_numeric(row.get("clip_frac"))
    if clip_frac is not None and clip_frac > 0.2:
        notes.append("High clipping fraction detected")
    grad_norm = _coerce_numeric(row.get("grad_norm"))
    if grad_norm is not None and grad_norm > 10.0:
        notes.append("Large gradient norm detected")
    kl_mean = _coerce_numeric(row.get("kl_mean"))
    if kl_mean is not None and kl_mean > 0.2:
        notes.append("High KL divergence detected")

    wall_time = _coerce_wall_time(row.get("wall_time")) or 0.0

    return Event(
        step=int(row["step"]),
        wall_time=wall_time,
        metrics=metrics,
        rng=rng,
        data_slice=data_slice,
        model_info=model_info,
        notes=notes,
    )


def events_to_dataframe(events: Sequence[Event | Dict[str, Any]]) -> pd.DataFrame:
    """Convert normalized events into a TrainingMetrics-style DataFrame."""

    rows: List[Dict[str, Any]] = []
    skipped = 0

    for event in events:
        if isinstance(event, Event):
            event_dict = {
                "step": event.step,
                "wall_time": event.wall_time,
                "metrics": event.metrics,
                "rng": event.rng,
                "data_slice": event.data_slice,
                "model_info": event.model_info,
            }
        elif isinstance(event, dict):
            event_dict = event
        else:
            skipped += 1
            continue

        step_value = event_dict.get("step")
        try:
            step_numeric = int(step_value)
        except (TypeError, ValueError):
            skipped += 1
            continue

        row: Dict[str, Any] = {"step": step_numeric}

        wall_time = _coerce_wall_time(event_dict.get("wall_time"))
        if wall_time is not None:
            row["wall_time"] = wall_time

        model_info = event_dict.get("model_info") or {}
        run_id = model_info.get("run_id")
        if run_id is not None and not _is_missing(run_id):
            row["run_id"] = str(run_id)
        phase = model_info.get("phase")
        if phase is not None and not _is_missing(phase):
            row["phase"] = phase
        git_sha = model_info.get("git_sha")
        if git_sha is not None and not _is_missing(git_sha):
            row["git_sha"] = git_sha

        rng = event_dict.get("rng") or {}
        seed = rng.get("seed")
        if seed is not None and not _is_missing(seed):
            row["seed"] = seed

        data_slice = event_dict.get("data_slice") or {}
        for key, value in data_slice.items():
            if _is_missing(value):
                continue
            numeric_value = _coerce_numeric(value)
            row[key] = numeric_value if numeric_value is not None else value

        metrics = event_dict.get("metrics") or {}
        for key, value in metrics.items():
            if _is_missing(value):
                continue
            numeric_value = _coerce_numeric(value)
            row[key] = numeric_value if numeric_value is not None else value

        rows.append(row)

    if skipped:
        logger.warning(
            "Skipped %d malformed event%s while converting to DataFrame",
            skipped,
            "" if skipped == 1 else "s",
        )

    if not rows:
        return pd.DataFrame(columns=_TRAINING_METRIC_COLUMNS)

    df = pd.DataFrame(rows)

    df["step"] = pd.to_numeric(df["step"], errors="coerce").astype("Int64")

    for column in _NUMERIC_COLUMNS.intersection(df.columns):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_values("step").reset_index(drop=True)
    return df.reindex(columns=_stable_column_order(df.columns))


def dataframe_to_events(
    df: pd.DataFrame, run_id: Optional[str] = None, git_sha: Optional[str] = None
) -> List[Event]:
    """Convert a TrainingMetrics-style DataFrame into normalized events."""

    if "step" not in df.columns:
        raise ValidationError(
            "DataFrame is missing required 'step' column",
            suggestion="Ensure every row includes a numeric 'step' value",
            error_code="MISSING_STEP_COLUMN",
        )

    standardized = df.copy()
    standardized["step"] = pd.to_numeric(standardized["step"], errors="coerce")
    invalid_steps = standardized["step"].isna()
    dropped = int(invalid_steps.sum())
    if dropped:
        logger.warning(
            "Dropped %d row%s without a valid step while converting DataFrame to events",
            dropped,
            "" if dropped == 1 else "s",
        )
        standardized = standardized.loc[~invalid_steps].copy()

    standardized = standardized.sort_values("step", kind="mergesort").reset_index(drop=True)

    events: List[Event] = []
    for _, row in standardized.iterrows():
        row_dict: Dict[str, Any] = {}
        for column, value in row.items():
            row_dict[column] = None if _is_missing(value) else value

        wall_time = _coerce_wall_time(row_dict.get("wall_time"))
        row_dict["wall_time"] = wall_time if wall_time is not None else 0.0

        row_run_id = row_dict.get("run_id") or run_id or "unknown"
        row_git_sha = row_dict.get("git_sha") or git_sha

        try:
            event = create_event_from_row(row_dict, str(row_run_id), row_git_sha)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning(
                "Skipping row at step %s due to error constructing event: %s",
                row_dict.get("step"),
                exc,
            )
            continue
        events.append(event)

    return events
