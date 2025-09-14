"""Reward model drift detection."""

import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, pearsonr, spearmanr

from rldk.io.readers import read_reward_head


def compare_models(
    model_a_dir: str, model_b_dir: str, prompts: List[str]
) -> Dict[str, Any]:
    """Compare two reward models and detect drift."""

    # Load reward models
    model_a_fn = read_reward_head(model_a_dir)
    model_b_fn = read_reward_head(model_b_dir)

    # Get scores
    scores_a = model_a_fn(prompts)
    scores_b = model_b_fn(prompts)

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

    return {
        "version": "1",
        "pearson": float(pearson_corr),
        "spearman": float(spearman_corr),
        "mae_z": float(mae_z),
        "l2_z": float(l2_z),
        "sign_flip_rate": float(sign_flip_rate),
        "slice_deltas": slice_deltas,
    }


def z_score(values: List[float]) -> np.ndarray:
    """Compute z-scores for a list of values."""
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return np.zeros_like(values)
    return (values - mean) / std


def compute_slice_deltas(
    prompts: List[str], scores_a: List[float], scores_b: List[float]
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
