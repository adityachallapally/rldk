"""Calibration analysis for reward models."""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def analyze_calibration(
    run_data: pd.DataFrame, reward_col: str, threshold: float = 0.7
) -> Tuple[float, Dict[str, Any]]:
    """
    Analyze reward model calibration against human preferences or ground truth.

    Args:
        run_data: Training run data
        reward_col: Column name for reward values
        threshold: Minimum acceptable calibration score

    Returns:
        Tuple of (calibration_score, calibration_details)
    """
    details = {}

    # Check if we have human preference data
    if "human_preference" in run_data.columns:
        score, pref_details = _analyze_human_preference_calibration(
            run_data, reward_col
        )
        details["human_preference"] = pref_details
    elif "ground_truth" in run_data.columns:
        score, gt_details = _analyze_ground_truth_calibration(run_data, reward_col)
        details["ground_truth"] = gt_details
    else:
        # No calibration data available
        score = 0.0
        details["error"] = "No human preference or ground truth data available"

    details["overall_score"] = score
    details["threshold"] = threshold
    details["passed"] = score >= threshold

    return score, details


def _analyze_human_preference_calibration(
    run_data: pd.DataFrame, reward_col: str
) -> Tuple[float, Dict[str, Any]]:
    """Analyze calibration against human preference scores."""
    details = {}

    # Get human preference and reward scores
    human_scores = run_data["human_preference"].values
    reward_scores = run_data[reward_col].values

    # Remove any NaN values
    valid_mask = ~(np.isnan(human_scores) | np.isnan(reward_scores))
    if not valid_mask.any():
        return 0.0, {"error": "No valid data for calibration analysis"}

    human_scores = human_scores[valid_mask]
    reward_scores = reward_scores[valid_mask]

    # Calculate correlation
    correlation = np.corrcoef(human_scores, reward_scores)[0, 1]
    details["correlation"] = correlation

    # Calculate calibration curve
    try:
        # Bin the data for calibration analysis
        n_bins = min(10, len(human_scores) // 10)
        if n_bins < 2:
            return 0.0, {"error": "Insufficient data for calibration analysis"}

        # Create bins based on reward scores
        bin_edges = np.percentile(reward_scores, np.linspace(0, 100, n_bins + 1))
        bin_indices = np.digitize(reward_scores, bin_edges)

        # Calculate mean human preference for each bin
        bin_means = []
        bin_counts = []
        for i in range(1, n_bins + 1):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_means.append(human_scores[mask].mean())
                bin_counts.append(mask.sum())

        if len(bin_means) < 2:
            return 0.0, {"error": "Insufficient bins for calibration analysis"}

        # Calculate calibration error (how well predicted probabilities match observed frequencies)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_centers = bin_centers[: len(bin_means)]

        # Normalize reward scores to [0, 1] for calibration
        reward_normalized = (reward_scores - reward_scores.min()) / (
            reward_scores.max() - reward_scores.min()
        )

        # Calculate Brier score (lower is better)
        brier_score = brier_score_loss(human_scores, reward_normalized)
        details["brier_score"] = brier_score

        # Calculate calibration score based on correlation and Brier score
        # Higher correlation and lower Brier score = better calibration
        calibration_score = (correlation + (1 - brier_score)) / 2
        calibration_score = max(0.0, min(1.0, calibration_score))  # Clamp to [0, 1]

        details["bin_centers"] = bin_centers.tolist()
        details["bin_means"] = bin_means
        details["bin_counts"] = bin_counts
        details["calibration_score"] = calibration_score

        return calibration_score, details

    except Exception as e:
        details["error"] = f"Calibration analysis failed: {str(e)}"
        return 0.0, details


def _analyze_ground_truth_calibration(
    run_data: pd.DataFrame, reward_col: str
) -> Tuple[float, Dict[str, Any]]:
    """Analyze calibration against ground truth labels."""
    details = {}

    # Get ground truth and reward scores
    ground_truth = run_data["ground_truth"].values
    reward_scores = run_data[reward_col].values

    # Remove any NaN values
    valid_mask = ~(np.isnan(ground_truth) | np.isnan(reward_scores))
    if not valid_mask.any():
        return 0.0, {"error": "No valid data for calibration analysis"}

    ground_truth = ground_truth[valid_mask]
    reward_scores = reward_scores[valid_mask]

    # Calculate correlation
    correlation = np.corrcoef(ground_truth, reward_scores)[0, 1]
    details["correlation"] = correlation

    # For binary ground truth, calculate AUC and calibration
    if len(np.unique(ground_truth)) == 2:
        # Binary classification case
        from sklearn.metrics import average_precision_score, roc_auc_score

        try:
            auc_score = roc_auc_score(ground_truth, reward_scores)
            ap_score = average_precision_score(ground_truth, reward_scores)

            details["auc_score"] = auc_score
            details["average_precision"] = ap_score

            # Calculate calibration score based on correlation and AUC
            calibration_score = (correlation + auc_score) / 2
            calibration_score = max(0.0, min(1.0, calibration_score))

            details["calibration_score"] = calibration_score
            return calibration_score, details

        except Exception as e:
            details["error"] = f"Binary calibration analysis failed: {str(e)}"
            return 0.0, details

    else:
        # Multi-class or regression case
        # Use correlation as the main calibration metric
        calibration_score = (correlation + 1) / 2  # Convert from [-1, 1] to [0, 1]
        details["calibration_score"] = calibration_score

        return calibration_score, details


def calculate_calibration_metrics(
    reward_scores: np.ndarray, true_labels: np.ndarray, n_bins: int = 10
) -> Dict[str, Any]:
    """
    Calculate detailed calibration metrics.

    Args:
        reward_scores: Predicted reward scores
        true_labels: True labels or human preferences
        n_bins: Number of bins for calibration analysis

    Returns:
        Dictionary with calibration metrics
    """
    try:
        # Normalize reward scores to [0, 1]
        scores_normalized = (reward_scores - reward_scores.min()) / (
            reward_scores.max() - reward_scores.min()
        )

        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            true_labels, scores_normalized, n_bins=n_bins
        )

        # Calculate Brier score
        brier_score = brier_score_loss(true_labels, scores_normalized)

        # Calculate ECE (Expected Calibration Error)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(scores_normalized, bin_edges)

        ece = 0.0
        for i in range(1, n_bins + 1):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_conf = scores_normalized[mask].mean()
                bin_acc = true_labels[mask].mean()
                bin_size = mask.sum()
                ece += (bin_size / len(true_labels)) * abs(bin_conf - bin_acc)

        return {
            "fraction_of_positives": fraction_of_positives.tolist(),
            "mean_predicted_value": mean_predicted_value.tolist(),
            "brier_score": brier_score,
            "ece": ece,
            "correlation": np.corrcoef(reward_scores, true_labels)[0, 1],
        }

    except Exception as e:
        return {"error": f"Calibration metrics calculation failed: {str(e)}"}
