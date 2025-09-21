"""Evaluation probes for different aspects of model performance."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..forensics.kl_schedule_tracker import KLScheduleTracker


def evaluate_alignment(data: pd.DataFrame, seed: int = 42, **kwargs) -> Dict[str, Any]:
    """
    Evaluate how well the model follows human preferences and instructions.

    Args:
        data: Training run data
        seed: Random seed for reproducibility
        **kwargs: Additional arguments

    Returns:
        Dictionary with alignment score and details
    """

    np.random.seed(seed)

    # Check if we have alignment-related metrics
    alignment_metrics = []

    # Check for reward alignment with human preferences
    if "human_preference" in data.columns and "reward_mean" in data.columns:
        # Calculate correlation between rewards and human preferences
        valid_mask = ~(data["human_preference"].isna() | data["reward_mean"].isna())
        if valid_mask.sum() > 10:
            human_prefs = data.loc[valid_mask, "human_preference"]
            rewards = data.loc[valid_mask, "reward_mean"]
            correlation = np.corrcoef(human_prefs, rewards)[0, 1]
            alignment_metrics.append(("human_preference_correlation", correlation))

    # Check for instruction following (if we have instruction compliance data)
    if "instruction_compliance" in data.columns:
        compliance_score = data["instruction_compliance"].mean()
        alignment_metrics.append(("instruction_compliance", compliance_score))

    # Check for safety adherence (if we have safety metrics)
    if "safety_score" in data.columns:
        safety_score = data["safety_score"].mean()
        alignment_metrics.append(("safety_adherence", safety_score))

    # Calculate overall alignment score
    if alignment_metrics:
        scores = [score for _, score in alignment_metrics]
        overall_score = np.mean(scores)
    else:
        # No metrics available - return None instead of default
        overall_score = None

    return {
        "score": float(overall_score) if overall_score is not None else np.nan,
        "details": f"Alignment evaluation based on {len(alignment_metrics)} metrics" if alignment_metrics else "No alignment metrics could be computed",
        "method": "correlation_and_consistency",
        "metrics": alignment_metrics,
        "sample_size": len(data),
    }


def evaluate_helpfulness(
    data: pd.DataFrame, seed: int = 42, **kwargs
) -> Dict[str, Any]:
    """
    Evaluate the quality and utility of model responses.

    Args:
        data: Training run data
        seed: Random seed for reproducibility
        **kwargs: Additional arguments

    Returns:
        Dictionary with helpfulness score and details
    """

    np.random.seed(seed)

    helpfulness_metrics = []

    # Check for response quality metrics
    if "response_quality" in data.columns:
        quality_score = data["response_quality"].mean()
        helpfulness_metrics.append(("response_quality", quality_score))

    # Check for relevance scores
    if "relevance_score" in data.columns:
        relevance_score = data["relevance_score"].mean()
        helpfulness_metrics.append(("relevance", relevance_score))

    # Check for completeness scores
    if "completeness_score" in data.columns:
        completeness_score = data["completeness_score"].mean()
        helpfulness_metrics.append(("completeness", completeness_score))

    # Check for user satisfaction (if available)
    if "user_satisfaction" in data.columns:
        satisfaction_score = data["user_satisfaction"].mean()
        helpfulness_metrics.append(("user_satisfaction", satisfaction_score))

    # Calculate overall helpfulness score
    if helpfulness_metrics:
        scores = [score for _, score in helpfulness_metrics]
        overall_score = np.mean(scores)
    else:
        # No metrics available - return None instead of default
        overall_score = None

    return {
        "score": float(overall_score) if overall_score is not None else np.nan,
        "details": f"Helpfulness evaluation based on {len(helpfulness_metrics)} metrics" if helpfulness_metrics else "No helpfulness metrics could be computed",
        "method": "quality_metrics_and_rewards",
        "metrics": helpfulness_metrics,
        "sample_size": len(data),
    }


def evaluate_harmlessness(
    data: pd.DataFrame, seed: int = 42, **kwargs
) -> Dict[str, Any]:
    """
    Evaluate model safety and harmlessness.

    Args:
        data: Training run data
        seed: Random seed for reproducibility
        **kwargs: Additional arguments

    Returns:
        Dictionary with harmlessness score and details
    """

    np.random.seed(seed)

    safety_metrics = []

    # Check for explicit safety scores
    if "safety_score" in data.columns:
        safety_score = data["safety_score"].mean()
        safety_metrics.append(("safety_score", safety_score))

    # Check for toxicity scores (lower is better)
    if "toxicity_score" in data.columns:
        toxicity_score = data["toxicity_score"].mean()
        # Invert toxicity score (lower toxicity = higher safety)
        inverted_toxicity = 1 - toxicity_score
        safety_metrics.append(("toxicity_inverted", inverted_toxicity))

    # Check for bias scores
    if "bias_score" in data.columns:
        bias_score = data["bias_score"].mean()
        # Invert bias score (lower bias = higher safety)
        inverted_bias = 1 - bias_score
        safety_metrics.append(("bias_inverted", inverted_bias))

    # Check for adversarial robustness
    if "adversarial_score" in data.columns:
        adv_score = data["adversarial_score"].mean()
        safety_metrics.append(("adversarial_robustness", adv_score))

    # Calculate overall harmlessness score
    if safety_metrics:
        scores = [score for _, score in safety_metrics]
        overall_score = np.mean(scores)
    else:
        # No metrics available - return None instead of default
        overall_score = None

    return {
        "score": float(overall_score) if overall_score is not None else np.nan,
        "details": f"Harmlessness evaluation based on {len(safety_metrics)} metrics" if safety_metrics else "No harmlessness metrics could be computed",
        "method": "safety_metrics_and_stability",
        "metrics": safety_metrics,
        "sample_size": len(data),
    }


def evaluate_hallucination(
    data: pd.DataFrame, seed: int = 42, **kwargs
) -> Dict[str, Any]:
    """
    Evaluate factual accuracy and hallucination detection.

    Args:
        data: Training run data
        seed: Random seed for reproducibility
        **kwargs: Additional arguments

    Returns:
        Dictionary with hallucination score and details (lower is better)
    """

    np.random.seed(seed)

    accuracy_metrics = []

    # Check for factual accuracy scores
    if "factual_accuracy" in data.columns:
        accuracy_score = data["factual_accuracy"].mean()
        accuracy_metrics.append(("factual_accuracy", accuracy_score))

    # Check for hallucination detection scores
    if "hallucination_score" in data.columns:
        hallucination_score = data["hallucination_score"].mean()
        # Invert hallucination score (lower hallucination = better)
        inverted_hallucination = 1 - hallucination_score
        accuracy_metrics.append(("hallucination_inverted", inverted_hallucination))

    # Check for citation accuracy
    if "citation_accuracy" in data.columns:
        citation_score = data["citation_accuracy"].mean()
        accuracy_metrics.append(("citation_accuracy", citation_score))

    # Check for consistency scores
    if "consistency_score" in data.columns:
        consistency_score = data["consistency_score"].mean()
        accuracy_metrics.append(("consistency", consistency_score))

    # Calculate overall hallucination score (lower is better)
    if accuracy_metrics:
        scores = [score for _, score in accuracy_metrics]
        overall_score = 1 - np.mean(scores)  # Invert so lower = better
    else:
        # No metrics available - return None instead of default
        overall_score = None

    return {
        "score": float(overall_score) if overall_score is not None else np.nan,
        "details": f"Hallucination evaluation based on {len(accuracy_metrics)} metrics" if accuracy_metrics else "No hallucination metrics could be computed",
        "method": "accuracy_metrics_and_consistency",
        "metrics": accuracy_metrics,
        "sample_size": len(data),
        "note": "Lower scores indicate better performance (less hallucination)",
    }


def evaluate_kl_drift(
    data: pd.DataFrame,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Evaluate KL drift severity using the schedule tracker."""

    if data.empty:
        return {
            "score": np.nan,
            "drift_score": np.nan,
            "detected": False,
            "trend": "unknown",
            "sample_size": 0,
            "details": "No data available for KL drift evaluation",
        }

    kl_candidates: List[str] = [
        kwargs.get("kl_column", "kl"),
        "kl",
        "kl_mean",
        "ppo/policy/kl_mean",
        "train/kl",
    ]
    step_candidates: List[str] = [kwargs.get("step_column", "step"), "global_step"]
    coef_candidates: List[str] = [
        kwargs.get("kl_coef_column", "kl_coef"),
        "kl_coef",
        "kl_coefficient",
        "kl_coeff",
        "ppo/policy/kl_coef",
    ]

    def _resolve_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        for candidate in candidates:
            if candidate in df.columns:
                return candidate
        return None

    step_col = _resolve_column(data, step_candidates)
    kl_col = _resolve_column(data, kl_candidates)
    if kl_col is None or step_col is None:
        return {
            "score": np.nan,
            "drift_score": np.nan,
            "detected": False,
            "trend": "unknown",
            "sample_size": len(data),
            "details": "Required KL or step columns not found for drift evaluation",
        }

    coef_col = _resolve_column(data, coef_candidates)
    sorted_data = data.sort_values(step_col)
    median_kl = sorted_data[kl_col].dropna().median()
    kl_target = float(median_kl) if not np.isnan(median_kl) else kwargs.get("kl_target", 0.1)

    tracker = KLScheduleTracker(
        kl_target=kl_target,
        kl_target_tolerance=kwargs.get("kl_target_tolerance", 0.05),
        window_size=kwargs.get("window_size", 200),
        drift_threshold=kwargs.get("drift_threshold", 0.15),
        drift_window_size=kwargs.get("drift_window_size", 100),
        reference_period=kwargs.get("reference_period", 500),
        enable_drift_tracking=True,
    )

    for _, row in sorted_data.iterrows():
        kl_value = row.get(kl_col)
        if pd.isna(kl_value):
            continue
        tracker.update(
            int(row.get(step_col, 0)),
            float(kl_value),
            float(row.get(coef_col, 1.0)) if coef_col else 1.0,
        )

    summary = tracker.get_summary()
    drift_score = summary.get("kl_drift_score", 0.0)
    detected = summary.get("kl_drift_detected", False)
    trend = summary.get("kl_drift_trend", "stable")

    return {
        "score": max(0.0, 1.0 - drift_score),
        "drift_score": drift_score,
        "detected": detected,
        "trend": trend,
        "reference_mean": summary.get("kl_reference_mean", 0.0),
        "reference_std": summary.get("kl_reference_std", 0.0),
        "current_mean": summary.get("kl_current_mean", 0.0),
        "current_std": summary.get("kl_current_std", 0.0),
        "sample_size": len(sorted_data),
        "details": "KL drift evaluation using KLScheduleTracker",
    }


def evaluate_kl_divergence(
    data: pd.DataFrame,
    reference_data: Optional[pd.DataFrame] = None,
    seed: int = 42,
    **kwargs,
) -> Dict[str, Any]:
    """
    Evaluate KL divergence between current run and reference distribution.

    Args:
        data: Current training run data
        reference_data: Reference data for comparison (if None, uses baseline)
        seed: Random seed for reproducibility
        **kwargs: Additional arguments including metrics to compare

    Returns:
        Dictionary with KL divergence scores and analysis
    """

    np.random.seed(seed)

    # Import KL divergence functions
    from .metrics import (
        calculate_kl_divergence_between_runs,
        calculate_kl_divergence_confidence_interval,
    )

    # Determine metrics to evaluate
    metrics_to_evaluate = kwargs.get(
        "metrics", ["reward_mean", "kl_mean", "entropy_mean"]
    )

    # If no reference data provided, create a synthetic baseline
    if reference_data is None:
        # Create baseline distribution based on expected behavior
        # This could be from a pre-trained model or theoretical expectations
        baseline_data = _create_baseline_distribution(data, metrics_to_evaluate, seed)
        reference_data = baseline_data
        reference_source = "synthetic_baseline"
    else:
        reference_source = "provided_reference"

    kl_results = {}
    overall_score = 0.0
    valid_metrics = 0

    for metric in metrics_to_evaluate:
        if metric not in data.columns:
            continue

        try:
            # Calculate KL divergence for this metric
            kl_result = calculate_kl_divergence_between_runs(
                reference_data, data, metric=metric
            )

            if "kl_divergence" in kl_result and not np.isnan(
                kl_result["kl_divergence"]
            ):
                kl_div = kl_result["kl_divergence"]

                # Convert KL divergence to a score (lower KL = higher score)
                # Use exponential decay: score = exp(-kl_div / scale_factor)
                scale_factor = 0.1  # Adjust based on expected KL divergence range
                metric_score = np.exp(-kl_div / scale_factor)

                kl_results[metric] = {
                    "kl_divergence": kl_div,
                    "score": metric_score,
                    "details": kl_result,
                }

                overall_score += metric_score
                valid_metrics += 1
            else:
                kl_results[metric] = {
                    "kl_divergence": np.nan,
                    "score": np.nan,
                    "error": kl_result.get("error", "Unknown error"),
                }

        except Exception as e:
            kl_results[metric] = {
                "kl_divergence": np.nan,
                "score": np.nan,
                "error": str(e),
            }

    # Calculate overall score
    if valid_metrics > 0:
        overall_score = overall_score / valid_metrics
    else:
        overall_score = 0.0

    # Calculate confidence intervals if we have sufficient data
    confidence_intervals = {}
    if valid_metrics > 0 and len(data) > 20:
        for metric in metrics_to_evaluate:
            if metric in kl_results and "kl_divergence" in kl_results[metric]:
                try:
                    ci_result = calculate_kl_divergence_confidence_interval(
                        reference_data, data, metric=metric
                    )
                    if "confidence_interval" in ci_result:
                        confidence_intervals[metric] = ci_result["confidence_interval"]
                except Exception:
                    confidence_intervals[metric] = (np.nan, np.nan)

    return {
        "score": float(overall_score),
        "kl_divergence_mean": float(
            np.mean(
                [
                    r.get("kl_divergence", np.nan)
                    for r in kl_results.values()
                    if not np.isnan(r.get("kl_divergence", np.nan))
                ]
            )
        ),
        "details": f"KL divergence evaluation across {valid_metrics} metrics",
        "method": "distribution_comparison",
        "reference_source": reference_source,
        "metrics_evaluated": list(kl_results.keys()),
        "kl_results": kl_results,
        "confidence_intervals": confidence_intervals,
        "sample_size": len(data),
        "reference_size": len(reference_data) if reference_data is not None else 0,
    }


def _create_baseline_distribution(
    data: pd.DataFrame, metrics: list, seed: int
) -> pd.DataFrame:
    """
    Create a synthetic baseline distribution for KL divergence comparison.

    Args:
        data: Current run data to base baseline on
        metrics: List of metrics to create baselines for
        seed: Random seed for reproducibility

    Returns:
        DataFrame with baseline distributions
    """
    np.random.seed(seed)

    baseline_data = pd.DataFrame()

    for metric in metrics:
        if metric in data.columns:
            # Get current metric values
            current_values = data[metric].dropna()

            if len(current_values) > 0:
                # Create baseline with similar distribution but slight differences
                # This simulates a "good" reference model
                mean_val = np.mean(current_values)
                std_val = np.std(current_values)

                # Add small random variation to create baseline
                baseline_values = np.random.normal(
                    mean_val,
                    std_val * 0.8,  # Slightly more concentrated
                    size=len(current_values),
                )

                # Ensure baseline values are in reasonable range
                if metric == "reward_mean":
                    baseline_values = np.clip(baseline_values, -10, 10)
                elif metric == "kl_mean":
                    baseline_values = np.clip(baseline_values, 0, 5)
                elif metric == "entropy_mean":
                    baseline_values = np.clip(baseline_values, 0, 10)

                baseline_data[metric] = baseline_values
            else:
                # If no data, create reasonable defaults
                if metric == "reward_mean":
                    baseline_data[metric] = np.random.normal(0, 1, size=100)
                elif metric == "kl_mean":
                    baseline_data[metric] = np.random.exponential(0.1, size=100)
                elif metric == "entropy_mean":
                    baseline_data[metric] = np.random.uniform(0, 5, size=100)
                else:
                    baseline_data[metric] = np.random.normal(0, 1, size=100)

    return baseline_data


def evaluate_reward_alignment(
    data: pd.DataFrame, seed: int = 42, **kwargs
) -> Dict[str, Any]:
    """
    Evaluate correlation between reward scores and human judgments.

    Args:
        data: Training run data
        seed: Random seed for reproducibility
        **kwargs: Additional arguments

    Returns:
        Dictionary with reward alignment score and details
    """

    np.random.seed(seed)

    alignment_metrics = []

    # Check for direct reward-human preference correlation
    if "human_preference" in data.columns and "reward_mean" in data.columns:
        valid_mask = ~(data["human_preference"].isna() | data["reward_mean"].isna())
        if valid_mask.sum() > 10:
            human_prefs = data.loc[valid_mask, "human_preference"]
            rewards = data.loc[valid_mask, "reward_mean"]

            # Calculate correlation
            correlation = np.corrcoef(human_prefs, rewards)[0, 1]
            alignment_metrics.append(("human_preference_correlation", correlation))

            # Calculate rank correlation (Spearman)
            try:
                spearman_corr = stats.spearmanr(human_prefs, rewards)[0]
                alignment_metrics.append(("spearman_correlation", spearman_corr))
            except (ValueError, TypeError) as e:
                # Skip if correlation calculation fails (e.g., constant values)
                print(f"Warning: Could not calculate Spearman correlation: {e}")
                pass

    # Check for reward calibration
    if "ground_truth" in data.columns and "reward_mean" in data.columns:
        valid_mask = ~(data["ground_truth"].isna() | data["reward_mean"].isna())
        if valid_mask.sum() > 10:
            ground_truth = data.loc[valid_mask, "ground_truth"]
            rewards = data.loc[valid_mask, "reward_mean"]

            # Calculate correlation with ground truth
            gt_correlation = np.corrcoef(ground_truth, rewards)[0, 1]
            alignment_metrics.append(("ground_truth_correlation", gt_correlation))

    # Check for reward consistency over time
    if "step" in data.columns and "reward_mean" in data.columns:
        # Sort by step and check for trends
        sorted_data = data.sort_values("step")
        rewards = sorted_data["reward_mean"].dropna()

        if len(rewards) > 10:
            # Calculate trend (slope of linear fit)
            steps = np.arange(len(rewards))
            if len(steps) > 1:
                slope = np.polyfit(steps, rewards, 1)[0]
                # Convert slope to stability score (lower absolute slope = more stable)
                stability_score = max(0, 1 - abs(slope))
                alignment_metrics.append(("reward_stability", stability_score))

    # Calculate overall reward alignment score
    if alignment_metrics:
        scores = [score for _, score in alignment_metrics]
        overall_score = np.mean(scores)
    else:
        # No metrics available - return None instead of default
        overall_score = None

    return {
        "score": float(overall_score) if overall_score is not None else np.nan,
        "details": f"Reward alignment evaluation based on {len(alignment_metrics)} metrics" if alignment_metrics else "No reward alignment metrics could be computed",
        "method": "correlation_and_stability",
        "metrics": alignment_metrics,
        "sample_size": len(data),
    }
