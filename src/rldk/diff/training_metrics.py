"""Utilities for comparing normalized training metrics tables."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import pandas as pd

from ..utils.error_handling import ValidationError


def _safe_float(value: Optional[float]) -> Optional[float]:
    """Return a finite float value or ``None`` if not representable."""

    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _format_available_columns(columns: Iterable[str]) -> str:
    """Format a helpful preview of available metric columns."""

    canonical = [column for column in columns if column != "step"]
    if not canonical:
        return "No metric columns available"
    preview = sorted(canonical)[:10]
    formatted = ", ".join(preview)
    if len(canonical) > 10:
        formatted += ", ..."
    return formatted


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric series grouped by step with non-null values."""

    numeric = pd.to_numeric(df[column], errors="coerce")
    grouped = numeric.groupby(df["step"]).mean()
    return grouped.dropna()


def _validate_signal_presence(df: pd.DataFrame, signal: str, label: str) -> None:
    """Ensure a signal exists within a normalized TrainingMetrics table."""

    if signal in df.columns:
        return

    available = _format_available_columns(df.columns)
    raise ValidationError(
        f"Signal '{signal}' not found in normalized run {label}",
        suggestion=(
            "Use --preset or --field-map to map your metric columns to canonical names. "
            f"Available columns: {available}"
        ),
        error_code="MISSING_SIGNAL",
        details={"missing_signal": signal, "run": label, "available_columns": available},
    )


def compare_training_metrics_tables(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    signals: List[str],
) -> Dict[str, object]:
    """Compare two TrainingMetrics tables for the requested signals.

    Args:
        df_a: Normalized TrainingMetrics DataFrame for run A.
        df_b: Normalized TrainingMetrics DataFrame for run B.
        signals: List of metric column names to compare.

    Returns:
        A JSON-serializable dictionary containing summary information and
        per-signal comparison statistics.

    Raises:
        ValidationError: If signals are missing or no overlap exists.
    """

    if not signals:
        raise ValidationError(
            "No signals were provided for diff comparison",
            suggestion="Pass at least one --signals option",
            error_code="NO_SIGNALS",
        )

    if df_a.empty:
        raise ValidationError(
            "Normalized run A contains no metrics",
            suggestion="Verify that run A has training metrics after normalization",
            error_code="EMPTY_RUN_A",
        )

    if df_b.empty:
        raise ValidationError(
            "Normalized run B contains no metrics",
            suggestion="Verify that run B has training metrics after normalization",
            error_code="EMPTY_RUN_B",
        )

    unique_signals: List[str] = []
    seen = set()
    for signal in signals:
        normalized = signal.strip()
        if not normalized:
            continue
        if normalized not in seen:
            unique_signals.append(normalized)
            seen.add(normalized)

    if not unique_signals:
        raise ValidationError(
            "Signals list resolves to zero valid entries",
            suggestion="Provide non-empty signal names with --signals",
            error_code="EMPTY_SIGNAL_LIST",
        )

    signal_results: List[Dict[str, object]] = []
    max_abs_delta: Optional[float] = None
    diff_count = 0
    min_shared_steps: Optional[int] = None
    max_shared_steps: Optional[int] = None
    total_steps_only_a = 0
    total_steps_only_b = 0

    for signal in unique_signals:
        _validate_signal_presence(df_a, signal, "A")
        _validate_signal_presence(df_b, signal, "B")

        series_a = _numeric_series(df_a, signal)
        series_b = _numeric_series(df_b, signal)

        steps_a = set(int(step) for step in series_a.index)
        steps_b = set(int(step) for step in series_b.index)

        shared_steps = sorted(steps_a & steps_b)
        steps_only_a = len(steps_a - steps_b)
        steps_only_b = len(steps_b - steps_a)

        total_steps_only_a += steps_only_a
        total_steps_only_b += steps_only_b

        status = "ok"
        comparison: Dict[str, object] = {
            "signal": signal,
            "steps_compared": len(shared_steps),
            "steps_only_a": steps_only_a,
            "steps_only_b": steps_only_b,
            "status": status,
        }

        if not shared_steps:
            comparison.update(
                {
                    "mean_a": None,
                    "mean_b": None,
                    "delta_mean": None,
                    "median_delta": None,
                    "max_delta": None,
                    "min_delta": None,
                    "max_abs_delta": None,
                }
            )
            comparison["status"] = "no_overlap"
            signal_results.append(comparison)
            continue

        aligned_a = series_a.loc[shared_steps]
        aligned_b = series_b.loc[shared_steps]
        deltas = aligned_b - aligned_a

        mean_a = _safe_float(aligned_a.mean())
        mean_b = _safe_float(aligned_b.mean())
        delta_mean = _safe_float(deltas.mean())
        median_delta = _safe_float(deltas.median())
        max_delta = _safe_float(deltas.max())
        min_delta = _safe_float(deltas.min())
        signal_max_abs = _safe_float(deltas.abs().max())

        if signal_max_abs is not None:
            if max_abs_delta is None or signal_max_abs > max_abs_delta:
                max_abs_delta = signal_max_abs
            if signal_max_abs > 0:
                status = "differs"
                diff_count += 1

        comparison.update(
            {
                "mean_a": mean_a,
                "mean_b": mean_b,
                "delta_mean": delta_mean,
                "median_delta": median_delta,
                "max_delta": max_delta,
                "min_delta": min_delta,
                "max_abs_delta": signal_max_abs,
                "status": status,
            }
        )

        shared_len = len(shared_steps)
        if min_shared_steps is None or shared_len < min_shared_steps:
            min_shared_steps = shared_len
        if max_shared_steps is None or shared_len > max_shared_steps:
            max_shared_steps = shared_len

        signal_results.append(comparison)

    summary: Dict[str, object] = {
        "requested_signals": unique_signals,
        "signals_compared": len(signal_results),
        "signals_with_differences": diff_count,
        "verdict": "differs" if diff_count else "match",
        "max_abs_delta": max_abs_delta,
        "steps_only_a": total_steps_only_a,
        "steps_only_b": total_steps_only_b,
    }

    if min_shared_steps is not None:
        summary["min_steps_compared"] = min_shared_steps
    if max_shared_steps is not None:
        summary["max_steps_compared"] = max_shared_steps

    return {"summary": summary, "signals": signal_results}


__all__ = ["compare_training_metrics_tables"]
