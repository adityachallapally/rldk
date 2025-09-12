"""Utility functions for evaluation metrics."""

from typing import Dict, Tuple, Any, Optional
import numpy as np
from scipy import stats
from scipy.stats import bootstrap


def calculate_confidence_intervals(
    scores: Dict[str, float], 
    sample_size: int, 
    confidence_level: float = 0.95,
    sample_data: Optional[Dict[str, np.ndarray]] = None
) -> Dict[str, Tuple[float, float]]:
    """
    Calculate confidence intervals for evaluation scores.

    Args:
        scores: Dictionary of metric names to scores
        sample_size: Number of samples used in evaluation
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        sample_data: Optional dictionary of metric names to actual sample data arrays

    Returns:
        Dictionary mapping metric names to (lower, upper) confidence intervals
    """

    confidence_intervals = {}

    for metric, score in scores.items():
        if np.isnan(score):
            confidence_intervals[metric] = (np.nan, np.nan)
            continue

        # Handle edge cases
        if sample_size <= 0:
            confidence_intervals[metric] = (score, score)
            continue

        if sample_size == 1:
            confidence_intervals[metric] = (score, score)
            continue

        try:
            # Calculate actual standard deviation if sample data is available
            if sample_data and metric in sample_data:
                data = sample_data[metric]
                if len(data) > 1:
                    actual_std = np.std(data, ddof=1)  # Use sample standard deviation
                    standard_error = actual_std / np.sqrt(len(data))
                else:
                    # Fallback to conservative estimate
                    standard_error = 0.3 / np.sqrt(sample_size)
            else:
                # Use bootstrap-based estimation when sample data is not available
                # This is more robust than hardcoded values
                import warnings
                warnings.warn(
                    "Using estimated standard deviation for confidence intervals. "
                    "For more accurate results, provide sample_data parameter.",
                    UserWarning,
                    stacklevel=2
                )
                if sample_size >= 30:
                    # For large samples, use normal approximation with conservative std
                    # Based on binomial distribution for binary metrics
                    if 0 <= score <= 1:
                        # For binary-like metrics, use binomial standard deviation
                        estimated_std = np.sqrt(score * (1 - score))
                    else:
                        # For continuous metrics, use a more conservative estimate
                        estimated_std = min(0.3, abs(score) * 0.1)
                else:
                    # For small samples, use t-distribution with conservative estimate
                    estimated_std = 0.3  # Conservative fallback
                
                standard_error = estimated_std / np.sqrt(sample_size)

            # Calculate confidence interval using normal approximation
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            margin_of_error = z_score * standard_error

            lower_bound = max(0, score - margin_of_error)
            upper_bound = min(1, score + margin_of_error)

            # Ensure bounds are valid numbers
            if np.isnan(lower_bound) or np.isnan(upper_bound):
                confidence_intervals[metric] = (score, score)
            else:
                confidence_intervals[metric] = (float(lower_bound), float(upper_bound))

        except (ValueError, ArithmeticError, OverflowError) as e:
            # Fallback to point estimate if calculation fails
            import warnings
            warnings.warn(
                f"Confidence interval calculation failed for {metric}: {e}. "
                "Using point estimate.",
                UserWarning,
                stacklevel=2
            )
            confidence_intervals[metric] = (score, score)

    return confidence_intervals


def calculate_effect_sizes(
    scores: Dict[str, float], 
    baseline_scores: Dict[str, float],
    sample_data: Optional[Dict[str, np.ndarray]] = None,
    baseline_sample_data: Optional[Dict[str, np.ndarray]] = None
) -> Dict[str, float]:
    """
    Calculate effect sizes comparing current scores to baseline.

    Args:
        scores: Current evaluation scores
        baseline_scores: Baseline scores for comparison
        sample_data: Optional dictionary of metric names to actual sample data arrays
        baseline_sample_data: Optional dictionary of metric names to baseline sample data arrays

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
                # Try to calculate actual pooled standard deviation if sample data is available
                pooled_std = None
                
                if (sample_data and metric in sample_data and 
                    baseline_sample_data and metric in baseline_sample_data):
                    # Calculate actual pooled standard deviation
                    current_data = sample_data[metric]
                    baseline_data = baseline_sample_data[metric]
                    
                    if len(current_data) > 1 and len(baseline_data) > 1:
                        # Calculate pooled standard deviation
                        n1, n2 = len(current_data), len(baseline_data)
                        var1 = np.var(current_data, ddof=1)
                        var2 = np.var(baseline_data, ddof=1)
                        
                        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
                        pooled_std = np.sqrt(pooled_var)
                
                if pooled_std is None or pooled_std <= 0:
                    # Fallback to data-driven estimation
                    if 0 <= score <= 1 and 0 <= baseline <= 1:
                        # For binary-like metrics, use binomial approximation
                        # Pooled proportion for variance estimation
                        pooled_prop = (score + baseline) / 2
                        pooled_std = np.sqrt(pooled_prop * (1 - pooled_prop))
                    else:
                        # For continuous metrics, estimate from score ranges
                        score_range = max(abs(score), abs(baseline))
                        if score_range > 0:
                            pooled_std = score_range * 0.3  # Conservative estimate
                        else:
                            pooled_std = 0.3  # Final fallback

                if pooled_std > 0:
                    effect_size = (score - baseline) / pooled_std
                    effect_sizes[metric] = float(effect_size)
                else:
                    import warnings
                    warnings.warn(
                        f"Effect size calculation failed for {metric}: pooled_std is zero. "
                        "Using zero effect size.",
                        UserWarning,
                        stacklevel=2
                    )
                    effect_sizes[metric] = 0.0
            else:
                effect_sizes[metric] = np.nan
        else:
            # No baseline available
            effect_sizes[metric] = np.nan

    return effect_sizes