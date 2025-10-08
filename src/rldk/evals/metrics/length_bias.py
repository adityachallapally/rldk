"""Evaluation helpers for detecting reward length bias."""

import json
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...reward import LengthBiasDetector, LengthBiasMetrics

# Candidate column names that frequently appear in training logs
_DEFAULT_RESPONSE_COLUMNS: Tuple[Optional[str], ...] = (
    "response", "response_text", "completion", "output", "output_text"
)
_DEFAULT_REWARD_COLUMNS: Tuple[Optional[str], ...] = (
    "reward_mean",
    "reward",
    "score",
    "reward_score",
    "mean_reward",
)
_DEFAULT_LENGTH_COLUMNS: Tuple[Optional[str], ...] = (
    "tokens_out",
    "response_tokens",
    "response_length",
    "token_count",
    "length",
)


def resolve_length_bias_columns(
    data: pd.DataFrame,
    *,
    response_col: Optional[str] = None,
    reward_col: Optional[str] = None,
    length_col: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve column names for response, reward, and length inputs."""

    def _select_column(candidates: Iterable[Optional[str]]) -> Optional[str]:
        for column in candidates:
            if column and column in data.columns:
                return column
        return None

    resolved_response = _select_column((response_col, *_DEFAULT_RESPONSE_COLUMNS))
    resolved_reward = _select_column((reward_col, *_DEFAULT_REWARD_COLUMNS))
    resolved_length = _select_column((length_col, *_DEFAULT_LENGTH_COLUMNS))
    return resolved_response, resolved_reward, resolved_length


def prepare_length_bias_inputs(
    data: pd.DataFrame,
    *,
    response_col: Optional[str] = None,
    reward_col: Optional[str] = None,
    length_col: Optional[str] = None,
) -> Tuple[List[Any], List[float], Optional[List[Optional[float]]]]:
    """Prepare aligned response, reward, and length sequences for detection."""

    if data.empty:
        raise ValueError("Input data is empty; cannot evaluate length bias")

    response_name, reward_name, length_name = resolve_length_bias_columns(
        data,
        response_col=response_col,
        reward_col=reward_col,
        length_col=length_col,
    )

    if reward_name is None:
        raise ValueError(
            "No reward column found. Provide --reward-col or include a column such as "
            "'reward_mean' in the input data."
        )

    responses: List[Any]
    if response_name is not None:
        responses = data[response_name].tolist()
    else:
        responses = []

    rewards_raw = data[reward_name].tolist()
    rewards: List[float] = []
    for value in rewards_raw:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            rewards.append(np.nan)
        else:
            try:
                rewards.append(float(value))
            except (TypeError, ValueError):
                rewards.append(np.nan)

    lengths: Optional[List[Optional[float]]]
    if length_name is not None:
        raw_lengths = data[length_name].tolist()
        converted: List[Optional[float]] = []
        for entry in raw_lengths:
            if entry is None or (isinstance(entry, float) and np.isnan(entry)):
                converted.append(None)
            else:
                try:
                    converted.append(float(entry))
                except (TypeError, ValueError):
                    converted.append(None)
        lengths = converted
    else:
        lengths = None

    if not responses:
        # Provide placeholder responses so the detector can use explicit lengths
        responses = list(range(len(rewards)))

    if lengths is None and response_name is None:
        raise ValueError(
            "Unable to infer response lengths. Provide either response text or a "
            "length/token count column."
        )

    return responses, rewards, lengths


def length_bias_score_from_metrics(
    metrics: LengthBiasMetrics, *, threshold: float
) -> Tuple[float, bool, float]:
    """Compute evaluation score, pass flag, and severity from detector metrics."""

    severity = metrics.bias_severity if metrics.bias_severity is not None else 0.0
    severity = float(min(max(severity, 0.0), 1.0))
    score = float(max(0.0, min(1.0, 1.0 - severity)))
    passed = severity <= threshold
    return score, passed, severity


def evaluate_length_bias(
    data: pd.DataFrame,
    *,
    response_col: Optional[str] = None,
    reward_col: Optional[str] = None,
    length_col: Optional[str] = None,
    threshold: float = 0.35,
    sample_size: Optional[int] = None,
    seed: Optional[int] = None,
    tokenizer: Optional[Any] = None,
    **_: Any,
) -> Dict[str, Any]:
    """Evaluate reward length bias using :class:`LengthBiasDetector`."""

    if data is None:
        raise ValueError("DataFrame is required for length bias evaluation")

    if sample_size is not None and sample_size < 0:
        raise ValueError("sample_size must be non-negative")

    working_data = data
    if sample_size is not None and sample_size < len(working_data):
        working_data = working_data.sample(n=sample_size, random_state=seed).reset_index(
            drop=True
        )

    responses, rewards, lengths = prepare_length_bias_inputs(
        working_data,
        response_col=response_col,
        reward_col=reward_col,
        length_col=length_col,
    )

    detector = LengthBiasDetector(tokenizer=tokenizer)
    metrics = detector.analyze_length_bias(responses, rewards, lengths)

    score, passed, severity = length_bias_score_from_metrics(metrics, threshold=threshold)
    metrics_dict = json.loads(json.dumps(asdict(metrics), default=_json_default))

    result = {
        "score": score,
        "passed": passed,
        "severity": severity,
        "threshold": float(threshold),
        "response_count": int(metrics.response_count),
        "num_samples": int(metrics.valid_sample_count),
        "method": "length_bias_detector",
        "metrics": metrics_dict,
        "recommendations": metrics.recommendations,
    }

    if metrics.recommendations:
        result["details"] = "; ".join(metrics.recommendations)

    return result


def _json_default(value: Any) -> Any:
    """Coerce numpy and dataclass values into JSON-serializable types."""

    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")
