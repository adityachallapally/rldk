"""Helpers for normalizing user supplied training metrics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union

import pandas as pd

from ..utils.error_handling import ValidationError, validate_file_path
from .stream_normalizer import stream_jsonl_to_dataframe

logger = logging.getLogger(__name__)


TRAINING_METRIC_COLUMNS: list[str] = [
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
    "git_sha",
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

_STREAM_EXTENSIONS = {".jsonl", ".ndjson"}
_TABLE_EXTENSIONS = {".csv", ".tsv", ".parquet"}

_ADAPTER_CANONICAL_OVERRIDES = {
    "reward_mean": "reward",
    "reward_std": "reward_std",
    "kl_mean": "kl",
    "entropy_mean": "entropy",
}

_POST_INGEST_ALIAS_SOURCES = {
    "reward_mean": "reward",
    "kl_mean": "kl",
    "entropy_mean": "entropy",
}


def _stable_column_order(columns: Iterable[str]) -> list[str]:
    ordered = [column for column in TRAINING_METRIC_COLUMNS if column in columns]
    extras = sorted(column for column in columns if column not in ordered)
    return ordered + extras


def standardize_training_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame that follows the canonical TrainingMetrics schema."""

    if "step" not in df.columns:
        raise ValidationError(
            "DataFrame is missing required 'step' column",
            suggestion="Use --preset or --field-map to map your step column to 'step'",
            error_code="MISSING_STEP_COLUMN",
        )

    standardized = df.copy()
    standardized["step"] = pd.to_numeric(standardized["step"], errors="coerce")

    invalid_steps = standardized["step"].isna()
    invalid_count = int(invalid_steps.sum())
    if invalid_count:
        logger.warning(
            "Dropped %d row%s with missing or non-numeric step during normalization",
            invalid_count,
            "" if invalid_count == 1 else "s",
        )
        standardized = standardized.loc[~invalid_steps].copy()

    for column in TRAINING_METRIC_COLUMNS:
        if column not in standardized.columns:
            standardized[column] = None

    standardized["step"] = standardized["step"].astype("int64")

    for column in _NUMERIC_COLUMNS.intersection(standardized.columns):
        standardized[column] = pd.to_numeric(standardized[column], errors="coerce")

    standardized = standardized.sort_values("step").reset_index(drop=True)
    return standardized.loc[:, _stable_column_order(standardized.columns)]


def _rename_with_field_map(
    df: pd.DataFrame, field_map: Optional[Dict[str, str]]
) -> pd.DataFrame:
    if not field_map:
        return df

    rename_map = {
        source: target
        for source, target in field_map.items()
        if isinstance(source, str) and isinstance(target, str)
    }
    if not rename_map:
        return df

    renamed = df.copy()
    overlapping = set(rename_map).intersection(df.columns)
    if overlapping:
        renamed = renamed.rename(columns=rename_map)

    for canonical, alias in _POST_INGEST_ALIAS_SOURCES.items():
        if canonical not in field_map.values():
            continue
        if canonical in renamed.columns and not renamed[canonical].dropna().empty:
            continue
        if alias in renamed.columns:
            renamed[canonical] = renamed[alias]

    return renamed


def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t")
        if suffix == ".parquet":
            return pd.read_parquet(path)
    except Exception as exc:
        raise ValidationError(
            f"Failed to read metrics table from {path}",
            suggestion="Ensure the file contains valid tabular data",
            error_code="TABLE_READ_FAILED",
        ) from exc

    raise ValidationError(
        f"Unsupported table format: {suffix}",
        suggestion="Use .csv, .tsv, or .parquet files",
        error_code="UNSUPPORTED_TABLE_FORMAT",
    )


def _invert_field_map(field_map: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
    if not field_map:
        return None

    inverted: Dict[str, str] = {}
    for source, canonical in field_map.items():
        if not isinstance(source, str) or not isinstance(canonical, str):
            continue
        adapter_canonical = _ADAPTER_CANONICAL_OVERRIDES.get(canonical, canonical)
        # FlexibleDataAdapter expects canonical -> actual mappings
        if adapter_canonical in inverted and inverted[adapter_canonical] != source:
            logger.debug(
                "Multiple source fields provided for canonical column '%s'; using '%s'", 
                adapter_canonical,
                source,
            )
        inverted[adapter_canonical] = source

    return inverted or None


def normalize_training_metrics_source(
    source: Union[str, Path], field_map: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """Normalize an arbitrary run input into the TrainingMetrics schema."""

    field_map = field_map or None

    try:
        path_obj = validate_file_path(source, must_exist=True)
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(
            f"Failed to resolve path: {source}",
            suggestion="Provide a valid file or directory path",
            error_code="INVALID_RUN_PATH",
        ) from exc

    adapter_field_map = _invert_field_map(field_map)

    if path_obj.is_dir():
        from .ingest import ingest_runs

        raw_df = ingest_runs(path_obj, field_map=adapter_field_map)
    elif path_obj.is_file():
        suffix = path_obj.suffix.lower()
        if suffix in _STREAM_EXTENSIONS:
            raw_df = stream_jsonl_to_dataframe(path_obj, field_map=field_map)
        elif suffix in _TABLE_EXTENSIONS:
            raw_df = _load_table(path_obj)
        else:
            from .ingest import ingest_runs

            raw_df = ingest_runs(path_obj, field_map=adapter_field_map)
    else:
        raise ValidationError(
            f"Unsupported run path: {source}",
            suggestion="Provide a JSONL file, metrics table, or directory of logs",
            error_code="UNSUPPORTED_RUN_SOURCE",
        )

    renamed = _rename_with_field_map(raw_df, field_map)
    return standardize_training_metrics(renamed)


def normalize_training_metrics(
    source: Union[pd.DataFrame, Sequence[Mapping[str, Any]], str, Path],
    field_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Normalize training metrics regardless of source format."""

    if isinstance(source, pd.DataFrame):
        raw_df = source.copy()
    elif isinstance(source, (str, Path)):
        return normalize_training_metrics_source(source, field_map=field_map)
    elif isinstance(source, Sequence):
        try:
            records = list(source)
        except TypeError as exc:  # pragma: no cover - defensive guard
            raise ValidationError(
                "Unable to iterate over training metrics records",
                suggestion="Provide a pandas DataFrame, list of dicts, or a file path",
                error_code="INVALID_TRAINING_METRICS_ITERABLE",
            ) from exc

        if not records:
            raw_df = pd.DataFrame()
        else:
            invalid_record = next(
                (record for record in records if not isinstance(record, Mapping)), None
            )
            if invalid_record is not None:
                raise ValidationError(
                    "All metric records must be mappings (dict-like objects)",
                    suggestion="Ensure each item is a dict with step and metric values",
                    error_code="INVALID_METRIC_RECORD",
                    details={"invalid_record": invalid_record},
                )
            raw_df = pd.DataFrame(records)
    else:
        raise ValidationError(
            f"Unsupported training metrics input type: {type(source).__name__}",
            suggestion="Provide a pandas DataFrame, list of dicts, or a metrics file path",
            error_code="UNSUPPORTED_TRAINING_METRICS_INPUT",
            details={"received_type": type(source).__name__},
        )

    renamed = _rename_with_field_map(raw_df, field_map)
    return standardize_training_metrics(renamed)


__all__ = [
    "TRAINING_METRIC_COLUMNS",
    "normalize_training_metrics",
    "normalize_training_metrics_source",
    "standardize_training_metrics",
]

