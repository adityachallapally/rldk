"""Core evaluation metrics for RL Debug Kit."""

from typing import Any, Dict, Iterable, Tuple

import warnings
from typing import Any, Dict, Iterable

import numpy as np


_EXTREME_KL_WARNING = "KL divergence extremely large; capping to 1e6 for stability."

from .bias import evaluate_bias
from .catastrophic_forgetting import evaluate_catastrophic_forgetting
from .length_bias import (
    evaluate_length_bias,
    length_bias_score_from_metrics,
    prepare_length_bias_inputs,
    resolve_length_bias_columns,
)
from .throughput import evaluate_throughput
from .toxicity import evaluate_toxicity
from .utils import calculate_confidence_intervals, calculate_effect_sizes


def _validate_distribution(dist: Iterable[float]) -> np.ndarray:
    array = np.asarray(dist, dtype=np.float64).flatten()

    if array.size == 0:
        return np.zeros(1, dtype=np.float64)

    if np.isnan(array).any():
        raise ValueError("Input distributions contain NaN values")
    if np.isinf(array).any():
        raise ValueError("Input distributions contain infinite values")
    if (array < 0).any():
        raise ValueError("Probability distributions must be non-negative")

    return array


def calculate_kl_divergence(
    p: Iterable[float],
    q: Iterable[float],
    epsilon: float = 1e-12,
) -> float:
    """Robust KL divergence calculation with numeric safeguards."""

    p_arr = _validate_distribution(p)
    q_arr = _validate_distribution(q)

    if p_arr.size != q_arr.size:
        raise ValueError("Distributions must have the same length")

    p_sum = p_arr.sum()
    q_sum = q_arr.sum()

    if p_sum <= 0:
        return 0.0
    if q_sum <= 0:
        return float("inf")

    p_norm = p_arr / p_sum
    q_norm = q_arr / q_sum

    valid = p_norm > 0
    if not np.any(valid):
        return 0.0

    if np.any((q_norm == 0) & valid):
        warnings.warn(_EXTREME_KL_WARNING, RuntimeWarning, stacklevel=2)
        return 1e6

    q_valid = q_norm[valid]
    ratio = p_norm[valid] / np.clip(q_valid, epsilon, None)

    if np.any(q_valid < epsilon):
        warnings.warn(_EXTREME_KL_WARNING, RuntimeWarning, stacklevel=2)
        return 1e6

    kl = np.sum(p_norm[valid] * np.log(ratio))

    if not np.isfinite(kl):
        ratio = np.clip(ratio, epsilon, 1e12)
        kl = np.sum(p_norm[valid] * np.log(ratio))

    kl = float(max(0.0, kl))

    if kl > 1e6:
        warnings.warn(_EXTREME_KL_WARNING, RuntimeWarning, stacklevel=2)
        return 1e6

    return kl


def _extract_metric_values(data: Any, metric: str | None) -> np.ndarray:
    try:
        import pandas as pd  # type: ignore
    except ImportError:  # pragma: no cover - pandas optional in some environments
        pd = None

    if pd is not None and isinstance(data, pd.DataFrame):
        if metric is None:
            raise ValueError("metric must be provided when passing DataFrame inputs")
        values = data[metric].to_numpy(dtype=np.float64)
    else:
        values = np.asarray(data, dtype=np.float64).flatten()

    return values[~np.isnan(values)]


def calculate_kl_divergence_between_runs(
    run1: Any,
    run2: Any,
    metric: str | None = None,
    bins: int = 20,
) -> Dict[str, Any]:
    """Compute KL divergence between two sets of scalar observations."""

    data1 = _extract_metric_values(run1, metric)
    data2 = _extract_metric_values(run2, metric)

    if data1.size < 2 or data2.size < 2:
        return {
            "kl_divergence": np.nan,
            "error": "Insufficient data for KL divergence comparison",
            "reference_size": int(data1.size),
            "test_size": int(data2.size),
        }

    data_min = float(np.min([data1.min(), data2.min()]))
    data_max = float(np.max([data1.max(), data2.max()]))
    if data_max == data_min:
        data_max = data_min + 1.0

    hist_range = (data_min, data_max)
    counts1, edges = np.histogram(data1, bins=bins, range=hist_range)
    counts2, _ = np.histogram(data2, bins=bins, range=hist_range)

    kl_value = calculate_kl_divergence(counts1, counts2)

    return {
        "kl_divergence": float(kl_value),
        "bins": bins,
        "range": hist_range,
        "edges": edges.tolist(),
        "reference_counts": counts1.tolist(),
        "test_counts": counts2.tolist(),
    }


def calculate_kl_divergence_confidence_interval(
    data1: Any,
    data2: Any,
    metric: str = "reward_mean",
    confidence_level: float = 0.95,
    n_bootstrap: int = 500,
    bins: int = 20,
) -> Dict[str, Any]:
    """Estimate KL divergence with simple bootstrap confidence intervals."""

    rng = np.random.default_rng()
    values1 = _extract_metric_values(data1, metric)
    values2 = _extract_metric_values(data2, metric)

    if values1.size < 10 or values2.size < 10:
        return {
            "kl_divergence": np.nan,
            "confidence_interval": (np.nan, np.nan),
            "error": "Insufficient data for bootstrap confidence intervals",
        }

    baseline = calculate_kl_divergence_between_runs(values1, values2, bins=bins)["kl_divergence"]

    samples: list[float] = []
    for _ in range(n_bootstrap):
        boot1 = rng.choice(values1, size=values1.size, replace=True)
        boot2 = rng.choice(values2, size=values2.size, replace=True)
        sample = calculate_kl_divergence_between_runs(boot1, boot2, bins=bins)["kl_divergence"]
        if np.isfinite(sample):
            samples.append(sample)

    if not samples:
        return {
            "kl_divergence": baseline,
            "confidence_interval": (np.nan, np.nan),
            "error": "Bootstrap sampling did not produce finite KL estimates",
        }

    lower_q = (1 - confidence_level) / 2
    upper_q = 1 - lower_q
    lower = float(np.quantile(samples, lower_q))
    upper = float(np.quantile(samples, upper_q))

    return {
        "kl_divergence": float(baseline),
        "confidence_interval": (lower, upper),
        "bootstrap_samples": len(samples),
    }

__all__ = [
    "evaluate_throughput",
    "evaluate_toxicity",
    "evaluate_bias",
    "evaluate_catastrophic_forgetting",
    "evaluate_length_bias",
    "length_bias_score_from_metrics",
    "prepare_length_bias_inputs",
    "resolve_length_bias_columns",
    "calculate_confidence_intervals",
    "calculate_effect_sizes",
    "calculate_kl_divergence",
    "calculate_kl_divergence_between_runs",
    "calculate_kl_divergence_confidence_interval",
]
