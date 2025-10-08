"""Statistical metrics for evaluation results."""

import warnings
from typing import Any, Dict, Tuple

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


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic_func: callable,
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for a statistic.

    Args:
        data: Array of data points
        statistic_func: Function to calculate the statistic
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        n_bootstrap: Number of bootstrap samples

    Returns:
        Tuple of (lower_bound, upper_bound)
    """

    try:
        # Use scipy's bootstrap function
        bootstrap_result = bootstrap(
            (data,),
            statistic_func,
            n_resamples=n_bootstrap,
            confidence_level=confidence_level,
        )

        return bootstrap_result.confidence_interval
    except Exception:
        # Fallback to manual bootstrap if scipy bootstrap fails
        return _manual_bootstrap(data, statistic_func, confidence_level, n_bootstrap)


def _manual_bootstrap(
    data: np.ndarray,
    statistic_func: callable,
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
) -> Tuple[float, float]:
    """Manual bootstrap implementation as fallback."""

    n_samples = len(data)
    bootstrap_stats = []

    for _ in range(n_bootstrap):
        # Sample with replacement
        bootstrap_sample = np.random.choice(data, size=n_samples, replace=True)
        try:
            stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(stat)
        except Exception:
            continue

    if not bootstrap_stats:
        return (np.nan, np.nan)

    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(bootstrap_stats, lower_percentile)
    upper_bound = np.percentile(bootstrap_stats, upper_percentile)

    return (lower_bound, upper_bound)


def calculate_statistical_significance(
    group1_scores: np.ndarray, group2_scores: np.ndarray, test_type: str = "t_test"
) -> Dict[str, Any]:
    """
    Calculate statistical significance between two groups of scores.

    Args:
        group1_scores: Scores from first group
        group2_scores: Scores from second group
        test_type: Type of statistical test ('t_test', 'mann_whitney', 'ks_test')

    Returns:
        Dictionary with test results including p-value and statistic
    """

    if len(group1_scores) == 0 or len(group2_scores) == 0:
        return {
            "p_value": np.nan,
            "statistic": np.nan,
            "significant": False,
            "test_type": test_type,
            "error": "Empty data groups",
        }

    try:
        if test_type == "t_test":
            # Independent t-test
            statistic, p_value = stats.ttest_ind(group1_scores, group2_scores)
        elif test_type == "mann_whitney":
            # Mann-Whitney U test (non-parametric)
            statistic, p_value = stats.mannwhitneyu(
                group1_scores, group2_scores, alternative="two-sided"
            )
        elif test_type == "ks_test":
            # Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(group1_scores, group2_scores)
        else:
            raise ValueError(f"Unknown test type: {test_type}")

        # Determine significance
        significant = p_value < 0.05

        return {
            "p_value": float(p_value),
            "statistic": float(statistic),
            "significant": significant,
            "test_type": test_type,
            "group1_size": len(group1_scores),
            "group2_size": len(group2_scores),
            "group1_mean": float(np.mean(group1_scores)),
            "group2_mean": float(np.mean(group2_scores)),
        }

    except Exception as e:
        return {
            "p_value": np.nan,
            "statistic": np.nan,
            "significant": False,
            "test_type": test_type,
            "error": str(e),
        }


def calculate_reliability_metrics(
    scores: np.ndarray, n_splits: int = 5
) -> Dict[str, float]:
    """
    Calculate reliability metrics for evaluation scores.

    Args:
        scores: Array of evaluation scores
        n_splits: Number of splits for cross-validation

    Returns:
        Dictionary with reliability metrics
    """

    if len(scores) < n_splits * 2:
        return {
            "split_half_reliability": np.nan,
            "cronbach_alpha": np.nan,
            "standard_error": np.nan,
            "error": "Insufficient data for reliability analysis",
        }

    try:
        # Split-half reliability
        np.random.shuffle(scores)
        split_point = len(scores) // 2
        half1 = scores[:split_point]
        half2 = scores[split_point:]

        if len(half1) > 0 and len(half2) > 0:
            correlation = np.corrcoef(half1, half2)[0, 1]
            # Spearman-Brown correction
            split_half_reliability = (
                (2 * correlation) / (1 + correlation)
                if not np.isnan(correlation)
                else np.nan
            )
        else:
            split_half_reliability = np.nan

        # Cronbach's alpha (simplified - assumes items are parallel)
        if len(scores) > 1:
            scores_std = np.std(scores)
            if scores_std > 0:
                # Simplified Cronbach's alpha calculation
                n_items = len(scores)
                item_variances = np.var(scores)
                total_variance = np.var(scores)

                if total_variance > 0:
                    cronbach_alpha = (n_items / (n_items - 1)) * (
                        1 - item_variances / total_variance
                    )
                else:
                    cronbach_alpha = np.nan
            else:
                cronbach_alpha = np.nan
        else:
            cronbach_alpha = np.nan

        # Standard error of measurement
        if not np.isnan(split_half_reliability) and split_half_reliability > 0:
            standard_error = np.std(scores) * np.sqrt(1 - split_half_reliability)
        else:
            standard_error = np.nan

        return {
            "split_half_reliability": (
                float(split_half_reliability)
                if not np.isnan(split_half_reliability)
                else np.nan
            ),
            "cronbach_alpha": (
                float(cronbach_alpha) if not np.isnan(cronbach_alpha) else np.nan
            ),
            "standard_error": (
                float(standard_error) if not np.isnan(standard_error) else np.nan
            ),
        }

    except Exception as e:
        return {
            "split_half_reliability": np.nan,
            "cronbach_alpha": np.nan,
            "standard_error": np.nan,
            "error": str(e),
        }


def calculate_effect_size_interpretation(effect_size: float) -> str:
    """
    Interpret Cohen's d effect size.

    Args:
        effect_size: Cohen's d effect size value

    Returns:
        String interpretation of the effect size
    """

    effect_size = abs(effect_size)

    if effect_size < 0.2:
        return "negligible"
    elif effect_size < 0.5:
        return "small"
    elif effect_size < 0.8:
        return "medium"
    elif effect_size < 1.2:
        return "large"
    elif effect_size < 2.0:
        return "very large"
    else:
        return "huge"


def calculate_kl_divergence(
    p: np.ndarray, q: np.ndarray, epsilon: float = 1e-8
) -> float:
    """
    Calculate KL divergence D_KL(P||Q) between two probability distributions.

    Args:
        p: Probability distribution P (target/reference)
        q: Probability distribution Q (model under test)
        epsilon: Small value to avoid log(0) and numerical instability

    Returns:
        KL divergence value (non-negative)

    Note:
        - D_KL(P||Q) measures how much information is lost when Q is used to approximate P
        - Lower values indicate Q is closer to P
        - Not symmetric: D_KL(P||Q) ≠ D_KL(Q||P)
    """
    # Input validation: same length
    if len(p) != len(q):
        raise ValueError("Distributions must have the same length")

    # Convert to numpy arrays with higher precision
    p = np.array(p, dtype=np.float64)
    q = np.array(q, dtype=np.float64)

    # Input validation: finite values
    if np.any(np.isnan(p)) or np.any(np.isnan(q)):
        raise ValueError("Input distributions contain NaN values")

    if np.any(np.isinf(p)) or np.any(np.isinf(q)):
        raise ValueError("Input distributions contain infinite values")

    # Input validation: non-negative values
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("Probability distributions must be non-negative")

    # Handle all zero cases
    p_sum = np.sum(p)
    q_sum = np.sum(q)

    if p_sum == 0 and q_sum == 0:
        return 0.0  # Both distributions are zero

    if p_sum == 0:
        return 0.0  # KL divergence is 0 when P is zero everywhere

    if q_sum == 0:
        return float('inf')  # KL divergence is infinite when Q is zero but P is not

    # Normalize once as (x + eps) over (sum + len*eps)
    p_normalized = (p + epsilon) / (p_sum + len(p) * epsilon)
    q_normalized = (q + epsilon) / (q_sum + len(q) * epsilon)

    # Mask small p values to avoid numerical issues
    mask = p_normalized > epsilon

    if not np.any(mask):
        return 0.0

    # Calculate log ratio
    log_ratio = np.log(p_normalized[mask] / q_normalized[mask])

    # Check for numerical issues and fallback with larger epsilon if needed
    if np.any(np.isnan(log_ratio)) or np.any(np.isinf(log_ratio)):
        # Fallback with larger epsilon (1e-6)
        epsilon_large = 1e-6
        p_safe = (p + epsilon_large) / (p_sum + len(p) * epsilon_large)
        q_safe = (q + epsilon_large) / (q_sum + len(q) * epsilon_large)

        mask_large = p_safe > epsilon_large
        if not np.any(mask_large):
            return 0.0

        log_ratio = np.log(p_safe[mask_large] / q_safe[mask_large])
        kl_div = np.sum(p_safe[mask_large] * log_ratio)
    else:
        kl_div = np.sum(p_normalized[mask] * log_ratio)

    # Ensure non-negative result
    kl_div = max(0.0, float(kl_div))

    # Cap result to 1e6
    if kl_div > 1e6:
        warnings.warn(f"KL divergence value {kl_div} is extremely large, capping to 1e6", stacklevel=2)
        kl_div = 1e6

    return kl_div


def calculate_kl_divergence_between_runs(
    run1_data: np.ndarray,
    run2_data: np.ndarray,
    metric: str = "reward_mean",
    bins: int = 20,
    epsilon: float = 1e-8,
) -> Dict[str, Any]:
    """
    Calculate KL divergence between two training runs for a specific metric.

    Args:
        run1_data: Data from first run (reference)
        run2_data: Data from second run (model under test)
        metric: Metric column name to compare
        bins: Number of bins for histogram discretization
        epsilon: Small value to avoid log(0)

    Returns:
        Dictionary with KL divergence results and analysis
    """
    try:
        # Extract metric data
        if hasattr(run1_data, "columns") and metric in run1_data.columns:
            # DataFrame case
            data1 = run1_data[metric].dropna().values
            data2 = run2_data[metric].dropna().values
        else:
            # Array case
            data1 = np.array(run1_data).flatten()
            data2 = np.array(run2_data).flatten()

        # Handle empty or NaN data
        data1 = data1[~np.isnan(data1)]
        data2 = data2[~np.isnan(data2)]

        if len(data1) == 0 or len(data2) == 0:
            return {
                "kl_divergence": np.nan,
                "error": "Insufficient data for KL divergence calculation",
                "data1_size": len(data1),
                "data2_size": len(data2),
            }

        # Build histograms
        min_val = min(np.min(data1), np.min(data2))
        max_val = max(np.max(data1), np.max(data2))

        # Add small buffer to avoid edge cases
        buffer = (max_val - min_val) * 0.01
        min_val -= buffer
        max_val += buffer

        # Create histograms
        hist1, _ = np.histogram(
            data1, bins=bins, range=(min_val, max_val), density=True
        )
        hist2, _ = np.histogram(
            data2, bins=bins, range=(min_val, max_val), density=True
        )

        # Call the core KL divergence function
        kl_div = calculate_kl_divergence(hist1, hist2, epsilon)

        # Additional analysis
        mean_diff = np.mean(data2) - np.mean(data1)
        std_diff = np.std(data2) - np.std(data1)

        # Calculate Jensen-Shannon divergence (symmetric version)
        js_div = calculate_jensen_shannon_divergence(hist1, hist2, epsilon)

        return {
            "kl_divergence": float(kl_div),
            "jensen_shannon_divergence": float(js_div),
            "mean_difference": float(mean_diff),
            "std_difference": float(std_diff),
            "data1_size": len(data1),
            "data2_size": len(data2),
            "data1_mean": float(np.mean(data1)),
            "data2_mean": float(np.mean(data2)),
            "data1_std": float(np.std(data1)),
            "data2_std": float(np.std(data2)),
            "bins": bins,
            "metric": metric,
        }

    except Exception as e:
        return {
            "kl_divergence": np.nan,
            "error": str(e),
            "data1_size": len(run1_data) if hasattr(run1_data, "__len__") else 0,
            "data2_size": len(run2_data) if hasattr(run2_data, "__len__") else 0,
        }


def calculate_jensen_shannon_divergence(
    p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10
) -> float:
    """
    Calculate Jensen-Shannon divergence between two probability distributions.

    Args:
        p: Probability distribution P
        q: Probability distribution Q
        epsilon: Small value to avoid log(0)

    Returns:
        Jensen-Shannon divergence value (symmetric, bounded by log(2))

    Note:
        - JSD(P,Q) = 0.5 * [D_KL(P||M) + D_KL(Q||M)] where M = 0.5 * (P + Q)
        - Symmetric: JSD(P,Q) = JSD(Q,P)
        - Bounded: 0 ≤ JSD(P,Q) ≤ log(2)
        - Lower values indicate distributions are more similar
    """
    if len(p) != len(q):
        raise ValueError("Distributions must have the same length")

    # Ensure probabilities sum to 1 and handle edge cases
    p = np.array(p, dtype=np.float64)
    q = np.array(q, dtype=np.float64)

    # Normalize to probability distributions
    p = p / (np.sum(p) + epsilon)
    q = q / (np.sum(q) + epsilon)

    # Add small epsilon to avoid log(0)
    p = p + epsilon
    q = q + epsilon

    # Renormalize
    p = p / np.sum(p)
    q = q / np.sum(q)

    # Calculate midpoint distribution M = 0.5 * (P + Q)
    m = 0.5 * (p + q)

    # Calculate JSD = 0.5 * [D_KL(P||M) + D_KL(Q||M)]
    kl_pm = calculate_kl_divergence(p, m, epsilon)
    kl_qm = calculate_kl_divergence(q, m, epsilon)

    jsd = 0.5 * (kl_pm + kl_qm)

    return float(jsd)


def calculate_kl_divergence_confidence_interval(
    data1: np.ndarray,
    data2: np.ndarray,
    metric: str = "reward_mean",
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
) -> Dict[str, Any]:
    """
    Calculate KL divergence with bootstrap confidence intervals.

    Args:
        data1: Data from first run (reference)
        data2: Data from second run (model under test)
        metric: Metric column name to compare
        confidence_level: Confidence level for intervals
        n_bootstrap: Number of bootstrap samples

    Returns:
        Dictionary with KL divergence and confidence intervals
    """
    try:
        # Extract metric data
        if hasattr(data1, "columns") and metric in data1.columns:
            data1_vals = data1[metric].dropna().values
            data2_vals = data2[metric].dropna().values
        else:
            data1_vals = np.array(data1).flatten()
            data2_vals = np.array(data2).flatten()

        # Remove NaN values
        data1_vals = data1_vals[~np.isnan(data1_vals)]
        data2_vals = data2_vals[~np.isnan(data2_vals)]

        if len(data1_vals) < 10 or len(data2_vals) < 10:
            return {
                "kl_divergence": np.nan,
                "confidence_interval": (np.nan, np.nan),
                "error": "Insufficient data for bootstrap confidence intervals",
                "data1_size": len(data1_vals),
                "data2_size": len(data2_vals),
            }

        # Bootstrap KL divergence calculation
        kl_values = []
        for _ in range(n_bootstrap):
            # Bootstrap sample from both datasets
            boot1 = np.random.choice(data1_vals, size=len(data1_vals), replace=True)
            boot2 = np.random.choice(data2_vals, size=len(data2_vals), replace=True)

            # Calculate KL divergence for bootstrap sample
            try:
                kl_result = calculate_kl_divergence_between_runs(boot1, boot2, metric)
                if "kl_divergence" in kl_result and not np.isnan(
                    kl_result["kl_divergence"]
                ):
                    kl_values.append(kl_result["kl_divergence"])
            except Exception:
                continue

        if len(kl_values) < n_bootstrap // 2:
            return {
                "kl_divergence": np.nan,
                "confidence_interval": (np.nan, np.nan),
                "error": "Bootstrap failed to produce sufficient valid samples",
                "data1_size": len(data1_vals),
                "data2_size": len(data2_vals),
            }

        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_bound = np.percentile(kl_values, lower_percentile)
        upper_bound = np.percentile(kl_values, upper_percentile)

        # Calculate point estimate
        point_estimate = np.mean(kl_values)

        return {
            "kl_divergence": float(point_estimate),
            "confidence_interval": (float(lower_bound), float(upper_bound)),
            "bootstrap_samples": len(kl_values),
            "bootstrap_std": float(np.std(kl_values)),
            "confidence_level": confidence_level,
            "data1_size": len(data1_vals),
            "data2_size": len(data2_vals),
        }

    except Exception as e:
        return {
            "kl_divergence": np.nan,
            "confidence_interval": (np.nan, np.nan),
            "error": str(e),
            "data1_size": len(data1) if hasattr(data1, "__len__") else 0,
            "data2_size": len(data2) if hasattr(data2, "__len__") else 0,
        }


def calculate_power_analysis(
    effect_size: float, alpha: float = 0.05, power: float = 0.8
) -> Dict[str, Any]:
    """
    Calculate required sample size for statistical power analysis.

    Args:
        effect_size: Expected effect size (Cohen's d)
        alpha: Significance level
        power: Desired statistical power

    Returns:
        Dictionary with power analysis results
    """

    try:
        from statsmodels.stats.power import TTestPower

        # Create power analysis object
        power_analysis = TTestPower()

        # Calculate required sample size
        required_n = power_analysis.solve_power(
            effect_size=effect_size, alpha=alpha, power=power, alternative="two-sided"
        )

        return {
            "required_sample_size": int(required_n),
            "effect_size": effect_size,
            "alpha": alpha,
            "power": power,
            "test_type": "t-test",
        }

    except ImportError:
        # Fallback calculation if statsmodels not available
        # Simplified power analysis using normal approximation

        # Z-scores for alpha and power
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        # Required sample size (per group for two-sample t-test)
        required_n = 2 * ((z_alpha + z_beta) / effect_size) ** 2

        return {
            "required_sample_size": int(required_n),
            "effect_size": effect_size,
            "alpha": alpha,
            "power": power,
            "test_type": "t-test (approximate)",
            "note": "Approximate calculation - statsmodels recommended for accuracy",
        }
