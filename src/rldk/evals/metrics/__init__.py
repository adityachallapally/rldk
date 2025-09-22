"""Core evaluation metrics for RL Debug Kit."""

from functools import lru_cache
from importlib import import_module
from typing import Any

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


@lru_cache(maxsize=1)
def _legacy_metrics() -> Any:
    """Import the legacy ``rldk.evals.metrics`` module lazily."""

    return import_module("rldk.evals.metrics")


def calculate_kl_divergence(*args: Any, **kwargs: Any) -> Any:
    """Proxy ``calculate_kl_divergence`` to the legacy metrics module."""

    return _legacy_metrics().calculate_kl_divergence(*args, **kwargs)


def calculate_kl_divergence_between_runs(*args: Any, **kwargs: Any) -> Any:
    """Proxy ``calculate_kl_divergence_between_runs`` to the legacy metrics module."""

    return _legacy_metrics().calculate_kl_divergence_between_runs(*args, **kwargs)


def calculate_kl_divergence_confidence_interval(*args: Any, **kwargs: Any) -> Any:
    """Proxy ``calculate_kl_divergence_confidence_interval`` to the legacy metrics module."""

    return _legacy_metrics().calculate_kl_divergence_confidence_interval(*args, **kwargs)

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
