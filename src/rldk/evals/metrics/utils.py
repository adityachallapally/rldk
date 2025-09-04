"""Utility functions for evaluation metrics."""

from typing import Dict, Tuple, Any
import numpy as np
from scipy import stats
from scipy.stats import bootstrap


def calculate_confidence_intervals(
    scores: Dict[str, float], sample_size: int, confidence_level: float = 0.95
) -> Dict[str, Tuple[float, float]]:
    """
    Calculate confidence intervals for evaluation scores.

    Args:
        scores: Dictionary of metric names to scores
        sample_size: Number of samples used in evaluation
        confidence_level: Confidence level (e.g., 0.95 for 95%)

    Returns:
        Dictionary mapping metric names to (lower, upper) confidence intervals
    """

    confidence_intervals = {}

    for metric, score in scores.items():
        if np.isnan(score):
            confidence_intervals[metric] = (np.nan, np.nan)
            continue

        # For now, use a simple approach based on sample size
        # In practice, you might want to bootstrap or use more sophisticated methods

        # Standard error approximation (assuming score is roughly normal)
        # This is a simplified approach - in practice you'd want actual sampling data
        if sample_size > 1:
            # Use a conservative estimate of standard error
            # Assuming scores are roughly in [0, 1] range
            estimated_std = 0.3  # Conservative estimate
            standard_error = estimated_std / np.sqrt(sample_size)

            # Calculate confidence interval using normal approximation
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            margin_of_error = z_score * standard_error

            lower_bound = max(0, score - margin_of_error)
            upper_bound = min(1, score + margin_of_error)

            confidence_intervals[metric] = (lower_bound, upper_bound)
        else:
            confidence_intervals[metric] = (score, score)

    return confidence_intervals


def calculate_effect_sizes(
    scores: Dict[str, float], baseline_scores: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate effect sizes comparing current scores to baseline.

    Args:
        scores: Current evaluation scores
        baseline_scores: Baseline scores for comparison

    Returns:
        Dictionary mapping metric names to effect sizes (Cohen's d)
    """

    effect_sizes = {}

    for metric, score in scores.items():
        if np.isnan(score):
            effect_sizes[metric] = np.nan
            continue

        if metric in baseline_scores:
            baseline = baseline_scores[metric]
            if not np.isnan(baseline):
                # Calculate Cohen's d effect size
                # For now, use a simplified approach since we don't have full distributions

                # Assume a reasonable pooled standard deviation
                # In practice, you'd calculate this from actual data
                pooled_std = 0.3  # Conservative estimate for scores in [0, 1]

                if pooled_std > 0:
                    effect_size = (score - baseline) / pooled_std
                    effect_sizes[metric] = float(effect_size)
                else:
                    effect_sizes[metric] = 0.0
            else:
                effect_sizes[metric] = np.nan
        else:
            # No baseline available
            effect_sizes[metric] = np.nan

    return effect_sizes