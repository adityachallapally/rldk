"""Utilities for normalizing streamed JSONL events into training metrics."""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from ..monitor.presets import FIELD_MAP_PRESETS, get_field_map_preset
from ..utils.error_handling import ValidationError, validate_file_path

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

_META_PASSTHROUGH_FIELDS = {"phase", "seed"}


def _combine_field_maps(
    field_map: Optional[Dict[str, str]], preset: Optional[str]
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if preset:
        preset_mapping = get_field_map_preset(preset)
        if preset_mapping is None:
            available = ", ".join(sorted(FIELD_MAP_PRESETS.keys()))
            raise ValidationError(
                f"Unknown field map preset '{preset}'",
                suggestion=f"Use one of: {available}",
                error_code="UNKNOWN_FIELD_MAP_PRESET",
            )
        mapping.update(preset_mapping)
    if field_map:
        mapping.update(field_map)
    return mapping


def _coerce_step(value: Any, line_number: int, path: Path) -> int:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValidationError(
            f"Missing step value on line {line_number} of {path}",
            suggestion="Ensure every event includes a numeric 'step' field",
            error_code="MISSING_STEP",
        )
    try:
        step_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"Invalid step value '{value}' on line {line_number} of {path}",
            suggestion="Steps must be integers or numeric strings",
            error_code="INVALID_STEP",
        ) from exc
    if math.isnan(step_float) or math.isinf(step_float):
        raise ValidationError(
            f"Step value '{value}' on line {line_number} of {path} is not finite",
            suggestion="Provide a finite numeric value for 'step'",
            error_code="INVALID_STEP",
        )
    return int(step_float)


def _coerce_wall_time(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        value_float = float(value)
        if not math.isfinite(value_float):
            return None
        return value_float
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            logger.debug("Ignoring non-ISO timestamp value: %s", value)
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).timestamp()
    return None


def _first_non_null(values: Iterable[Any]) -> Optional[Any]:
    for item in values:
        if item is None:
            continue
        if isinstance(item, float) and (math.isnan(item) or math.isinf(item)):
            continue
        return item
    return None


def _stable_column_order(columns: Iterable[str]) -> list[str]:
    ordered: list[str] = [col for col in _TRAINING_METRIC_COLUMNS if col in columns]
    extras = sorted(col for col in columns if col not in ordered)
    return ordered + extras


def stream_jsonl_to_dataframe(
    path: str | Path,
    field_map: Optional[Dict[str, str]] = None,
    preset: Optional[str] = None,
) -> pd.DataFrame:
    """Normalize EventWriter-style JSONL streams into training metrics.

    Args:
        path: Path to the JSONL file containing event records.
        field_map: Optional explicit mapping from source keys to canonical keys.
        preset: Optional preset name that provides default field mappings.

    Returns:
        Pandas DataFrame sorted by step with TrainingMetrics-compatible columns.

    Raises:
        ValidationError: If the file path or required fields are invalid.
    """

    path_obj = validate_file_path(path, file_extensions=[".jsonl"])
    mapping = _combine_field_maps(field_map, preset)

    metrics = defaultdict(lambda: defaultdict(list))
    run_ids = defaultdict(list)
    wall_times = defaultdict(list)
    meta_fields = defaultdict(lambda: defaultdict(list))

    invalid_lines = 0
    valid_events = 0

    with path_obj.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                invalid_lines += 1
                continue
            if not isinstance(payload, dict):
                invalid_lines += 1
                continue
            mapped = {mapping.get(key, key): value for key, value in payload.items()}

            step = _coerce_step(mapped.get("step"), line_number, path_obj)
            name = mapped.get("name")
            value = mapped.get("value")

            if name is None:
                logger.debug("Skipping event without a name on line %d", line_number)
                continue
            if value is None:
                logger.debug("Skipping event without a value on line %d", line_number)
                continue

            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                logger.debug(
                    "Skipping event with non-numeric value %r on line %d", value, line_number
                )
                continue
            if math.isnan(numeric_value) or math.isinf(numeric_value):
                logger.debug(
                    "Skipping event with non-finite value %r on line %d",
                    value,
                    line_number,
                )
                continue

            metrics[step][str(name)].append(numeric_value)

            wall_time = _coerce_wall_time(mapped.get("time"))
            if wall_time is not None:
                wall_times[step].append(wall_time)

            run_id = mapped.get("run_id")
            if run_id is not None:
                run_ids[step].append(str(run_id))

            meta = mapped.get("meta")
            if isinstance(meta, dict):
                for key in _META_PASSTHROUGH_FIELDS:
                    if key in meta:
                        meta_fields[step][key].append(meta[key])

            valid_events += 1

    if invalid_lines:
        logger.warning(
            "Skipped %d invalid JSONL line%s while reading %s",
            invalid_lines,
            "s" if invalid_lines != 1 else "",
            path_obj,
        )

    if valid_events == 0 and invalid_lines == 0:
        logger.info("No events found in %s; returning empty TrainingMetrics frame", path_obj)
        return pd.DataFrame(columns=_TRAINING_METRIC_COLUMNS)

    if not metrics:
        if valid_events == 0:
            raise ValidationError(
                f"No valid events could be read from {path_obj}",
                suggestion="Ensure the file contains JSON objects with step, name, and value",
                error_code="NO_VALID_EVENTS",
            )
        return pd.DataFrame(columns=_TRAINING_METRIC_COLUMNS)

    rows = []
    for step in sorted(metrics.keys()):
        row: Dict[str, Any] = {"step": step}
        for metric_name, values in metrics[step].items():
            if not values:
                continue
            row[metric_name] = fmean(values)
        if wall_times[step]:
            row["wall_time"] = fmean(wall_times[step])
        run_id_value = _first_non_null(run_ids[step])
        if run_id_value is not None:
            row["run_id"] = run_id_value
        for key, values in meta_fields[step].items():
            meta_value = _first_non_null(values)
            if meta_value is not None:
                row[key] = meta_value
        rows.append(row)

    result = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    result = result.reindex(columns=_stable_column_order(result.columns), fill_value=None)
    return result


__all__ = ["stream_jsonl_to_dataframe"]

