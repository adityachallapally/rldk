"""Reward model drift detection."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, pearsonr, spearmanr, t

from rldk.io.readers import read_reward_head


def compare_models(
    model_a_dir: str, model_b_dir: str, prompts: Sequence[str]
) -> Dict[str, Any]:
    """Compare two reward models and detect drift."""

    prompts_list = list(prompts)
    if len(prompts_list) < 2:
        raise ValueError(
            "Reward drift comparison requires at least two prompts to compute statistics."
        )

    # Load reward models
    model_a_fn = read_reward_head(model_a_dir)
    model_b_fn = read_reward_head(model_b_dir)

    # Get scores
    scores_a = model_a_fn(prompts_list)
    scores_b = model_b_fn(prompts_list)

    return compare_score_lists(prompts_list, scores_a, scores_b)


def compare_score_lists(
    prompts: Sequence[str],
    scores_a: Iterable[float],
    scores_b: Iterable[float],
) -> Dict[str, Any]:
    """Compare two sets of scores that correspond to the same prompts."""

    scores_a_arr = np.asarray(list(scores_a), dtype=float)
    scores_b_arr = np.asarray(list(scores_b), dtype=float)

    if scores_a_arr.shape != scores_b_arr.shape:
        raise ValueError(
            "Score sequences must have the same length for drift comparison."
        )
    if scores_a_arr.size < 2:
        raise ValueError(
            "Reward drift comparison requires at least two scores on each side."
        )

    prompts_list = list(prompts)
    if prompts_list:
        if len(prompts_list) != scores_a_arr.size:
            raise ValueError(
                "Number of prompts must match the number of scores for drift comparison."
            )
    else:
        prompts_list = [""] * scores_a_arr.size

    return _compute_drift_report(prompts_list, scores_a_arr, scores_b_arr)


def _compute_drift_report(
    prompts: List[str], scores_a: np.ndarray, scores_b: np.ndarray
) -> Dict[str, Any]:
    """Compute the reward drift report for two aligned score arrays."""

    # Compute correlation metrics
    pearson_corr, _ = pearsonr(scores_a, scores_b)
    spearman_corr, _ = spearmanr(scores_a, scores_b)

    # Compute z-scored distances
    scores_a_z = z_score(scores_a)
    scores_b_z = z_score(scores_b)

    mae_z = np.mean(np.abs(scores_a_z - scores_b_z))
    l2_z = np.sqrt(np.mean((scores_a_z - scores_b_z) ** 2))

    # Compute sign flip rate
    sign_flip_rate = np.mean(np.sign(scores_a) != np.sign(scores_b))

    # Compute slice deltas
    slice_deltas = compute_slice_deltas(prompts, scores_a, scores_b)

    mean_diff, confidence_summary = _compute_confidence_summary(scores_a, scores_b)
    effect_size = _compute_effect_size(scores_a, scores_b)

    return {
        "version": "1",
        "pearson": float(pearson_corr),
        "spearman": float(spearman_corr),
        "mae_z": float(mae_z),
        "l2_z": float(l2_z),
        "sign_flip_rate": float(sign_flip_rate),
        "drift_magnitude": float(abs(mean_diff)),
        "effect_size": float(effect_size),
        "confidence_summary": confidence_summary,
        "slice_deltas": slice_deltas,
    }


def z_score(values: Sequence[float]) -> np.ndarray:
    """Compute z-scores for a list of values."""
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return np.zeros_like(values)
    return (values - mean) / std


def compute_slice_deltas(
    prompts: Sequence[str], scores_a: Sequence[float], scores_b: Sequence[float]
) -> Dict[str, Dict[str, Any]]:
    """Compute deltas for different prompt slices."""
    slice_deltas = {}

    # Define slice patterns
    slice_patterns = {
        "math": r"\b\d+\s*[\+\-\*\/]\s*\d+\b|\b(solve|calculate|compute|equation|formula)\b",
        "safety": r"\b(harm|danger|unsafe|illegal|kill|hurt|attack|weapon)\b",
        "refusal": r"\b(cannot|unable|sorry|apologize|decline|refuse|not allowed)\b",
        "code": r"\b(def|function|class|import|print|return|if|for|while)\b|```",
    }

    # Create slices
    for slice_name, pattern in slice_patterns.items():
        slice_indices = []
        for i, prompt in enumerate(prompts):
            if re.search(pattern, prompt, re.IGNORECASE):
                slice_indices.append(i)

        if slice_indices:
            slice_scores_a = [scores_a[i] for i in slice_indices]
            slice_scores_b = [scores_b[i] for i in slice_indices]

            delta_mean = np.mean(np.array(slice_scores_b) - np.array(slice_scores_a))

            slice_deltas[slice_name] = {
                "delta_mean": float(delta_mean),
                "n": len(slice_indices),
            }

    return slice_deltas


def detect_reward_drift(
    run_data: pd.DataFrame,
    reference_data: pd.DataFrame,
    reward_col: str = "reward_mean",
    step_col: str = "step",
    threshold_drift: float = 0.1,
) -> Tuple[bool, pd.DataFrame]:
    """
    Detect reward drift between run data and reference data.

    Args:
        run_data: Current run data
        reference_data: Reference run data for comparison
        reward_col: Column name for reward values
        step_col: Column name for training steps
        threshold_drift: Threshold for drift detection (KS test p-value)

    Returns:
        Tuple of (drift_detected, drift_metrics)
    """
    # Ensure both datasets have the required columns
    if reward_col not in run_data.columns:
        raise ValueError(f"Column '{reward_col}' not found in run_data")
    if reward_col not in reference_data.columns:
        raise ValueError(f"Column '{reward_col}' not found in reference_data")

    # Get reward values
    run_rewards = run_data[reward_col].dropna()
    ref_rewards = reference_data[reward_col].dropna()

    if len(run_rewards) == 0 or len(ref_rewards) == 0:
        return False, pd.DataFrame()

    # Perform KS test for drift detection
    ks_statistic, p_value = ks_2samp(run_rewards, ref_rewards)

    drift_detected = bool(p_value < threshold_drift)

    # Create drift metrics DataFrame
    drift_metrics = pd.DataFrame(
        {
            "metric": ["ks_statistic", "p_value", "drift_detected"],
            "value": [ks_statistic, p_value, drift_detected],
        }
    )

    return drift_detected, drift_metrics


def _compute_effect_size(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Compute Cohen's d effect size for two score distributions."""

    n_a = scores_a.size
    n_b = scores_b.size
    if n_a < 2 or n_b < 2:
        return 0.0

    var_a = np.var(scores_a, ddof=1)
    var_b = np.var(scores_b, ddof=1)
    pooled_denom = ((n_a - 1) * var_a + (n_b - 1) * var_b)
    if pooled_denom <= 0:
        return 0.0

    pooled_std = math.sqrt(pooled_denom / (n_a + n_b - 2))
    if pooled_std == 0 or not math.isfinite(pooled_std):
        return 0.0

    return float((np.mean(scores_b) - np.mean(scores_a)) / pooled_std)


def _compute_confidence_summary(
    scores_a: np.ndarray, scores_b: np.ndarray
) -> Tuple[float, str]:
    """Compute mean difference and a 95% confidence interval summary string."""

    mean_diff = float(np.mean(scores_b) - np.mean(scores_a))

    n_a = scores_a.size
    n_b = scores_b.size
    if n_a < 2 or n_b < 2:
        return mean_diff, "Not enough data to compute confidence interval."

    var_a = np.var(scores_a, ddof=1)
    var_b = np.var(scores_b, ddof=1)
    se_diff = math.sqrt(var_a / n_a + var_b / n_b)
    if se_diff == 0 or not math.isfinite(se_diff):
        return mean_diff, "Not enough variance to compute confidence interval."

    df = min(n_a, n_b) - 1
    if df <= 0:
        return mean_diff, "Not enough data to compute confidence interval."

    t_value = t.ppf(0.975, df)
    if not math.isfinite(t_value):
        return mean_diff, "Not enough data to compute confidence interval."

    lower = mean_diff - t_value * se_diff
    upper = mean_diff + t_value * se_diff
    summary = f"Mean difference {mean_diff:.4f} (95% CI [{lower:.4f}, {upper:.4f}])"
    return mean_diff, summary
