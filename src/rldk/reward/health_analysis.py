"""Main reward health checking functionality."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .calibration import analyze_calibration
from .drift import detect_reward_drift
from .length_bias import LengthBiasDetector, LengthBiasMetrics


@dataclass
class RewardHealthReport:
    """Comprehensive report of reward model health analysis."""

    passed: bool
    drift_detected: bool
    saturation_issues: List[str]
    calibration_score: float
    shortcut_signals: List[str]
    label_leakage_risk: float
    fixes: List[str]
    drift_metrics: pd.DataFrame
    calibration_details: Dict[str, Any]
    shortcut_analysis: Dict[str, float]
    saturation_analysis: Dict[str, Any]
    length_bias_detected: bool = False
    length_bias_metrics: LengthBiasMetrics = field(default_factory=LengthBiasMetrics)
    length_bias_recommendations: List[str] = field(default_factory=list)


def health(
    run_data: pd.DataFrame,
    reference_data: Optional[pd.DataFrame] = None,
    reward_col: str = "reward_mean",
    step_col: str = "step",
    threshold_drift: float = 0.1,
    threshold_saturation: float = 0.8,
    threshold_calibration: float = 0.7,
    threshold_shortcut: float = 0.6,
    threshold_leakage: float = 0.3,
    response_col: Optional[str] = None,
    length_col: Optional[str] = None,
    threshold_length_bias: float = 0.4,
    enable_length_bias_detection: bool = True,
) -> RewardHealthReport:
    """
    Analyze reward model health and detect pathologies.

    Args:
        run_data: Training run data with reward metrics
        reference_data: Optional reference run for comparison
        reward_col: Column name for reward values
        step_col: Column name for training steps
        threshold_drift: Threshold for drift detection (KS test p-value)
        threshold_saturation: Threshold for saturation detection
        threshold_calibration: Threshold for calibration quality
        threshold_shortcut: Threshold for shortcut signal detection
        threshold_leakage: Threshold for label leakage risk
        response_col: Optional column containing response text
        length_col: Optional column containing precomputed response lengths
        threshold_length_bias: Threshold on length bias severity for failures
        enable_length_bias_detection: Toggle for length bias detector

    Returns:
        RewardHealthReport with comprehensive analysis
    """
    # Validate input data
    if reward_col not in run_data.columns:
        raise ValueError(f"Reward column '{reward_col}' not found in run data")

    if step_col not in run_data.columns:
        raise ValueError(f"Step column '{step_col}' not found in run data")

    # Sort by step to ensure chronological order
    run_data = run_data.sort_values(step_col).reset_index(drop=True)

    # Initialize results
    issues = []
    fixes = []

    # 1. Detect reward drift
    drift_detected = False
    drift_metrics = pd.DataFrame()
    if reference_data is not None:
        drift_detected, drift_metrics = detect_reward_drift(
            run_data, reference_data, reward_col, step_col, threshold_drift
        )
        if drift_detected:
            issues.append("Reward drift detected compared to reference")
            fixes.append(
                "Check for data distribution shifts or model architecture changes"
            )

    # 2. Detect reward saturation
    saturation_issues = []
    saturation_analysis = {}
    if reward_col in run_data.columns:
        saturation_issues, saturation_analysis = _detect_saturation(
            run_data[reward_col], threshold_saturation
        )
        if saturation_issues:
            issues.extend(saturation_issues)
            fixes.append("Adjust reward scaling or check for gradient issues")

    # 3. Analyze calibration
    calibration_score = 0.0
    calibration_details = {}
    if "human_preference" in run_data.columns or "ground_truth" in run_data.columns:
        calibration_score, calibration_details = analyze_calibration(
            run_data, reward_col, threshold_calibration
        )
        if calibration_score < threshold_calibration:
            issues.append("Poor reward calibration detected")
            fixes.append("Retrain reward model with better human preference data")

    # 4. Detect shortcut signals
    shortcut_signals = []
    shortcut_analysis = {}
    shortcut_signals, shortcut_analysis = _detect_shortcut_signals(
        run_data, reward_col, threshold_shortcut
    )
    if shortcut_signals:
        issues.extend(shortcut_signals)
        fixes.append("Remove or balance shortcut features from training data")

    # 4b. Detect length bias using dedicated detector
    length_bias_detected = False
    length_bias_metrics = LengthBiasMetrics()
    length_bias_recommendations: List[str] = []
    if enable_length_bias_detection:
        responses, lengths = _prepare_length_bias_inputs(
            run_data, response_col, length_col
        )
        if responses is not None or lengths is not None:
            detector = LengthBiasDetector()
            rewards = run_data[reward_col].tolist()
            response_inputs: Iterable[Any]
            if responses is None:
                response_inputs = [""] * len(rewards)
            else:
                response_inputs = responses
            metrics = detector.analyze_length_bias(
                response_inputs,
                rewards,
                lengths,
            )
            length_bias_metrics = metrics
            length_bias_recommendations = list(metrics.recommendations)
            severity = metrics.bias_severity or 0.0
            if severity >= threshold_length_bias:
                length_bias_detected = True
                message = f"Length bias detected (severity {severity:.2f})"
                issues.append(message)
                shortcut_signals.append(message)
            for recommendation in metrics.recommendations:
                if recommendation and recommendation not in fixes:
                    fixes.append(recommendation)
        else:
            skip_message = (
                "Length bias detection skipped: no response or length data available."
            )
            length_bias_recommendations = [skip_message]
            if skip_message not in fixes:
                fixes.append(skip_message)

    # 5. Detect label leakage
    label_leakage_risk = _detect_label_leakage(run_data, reward_col, threshold_leakage)
    if label_leakage_risk > threshold_leakage:
        issues.append("Potential label leakage detected")
        fixes.append("Audit data pipeline for information leakage")

    # Determine overall health
    passed = len(issues) == 0

    return RewardHealthReport(
        passed=passed,
        drift_detected=drift_detected,
        saturation_issues=saturation_issues,
        calibration_score=calibration_score,
        shortcut_signals=shortcut_signals,
        label_leakage_risk=label_leakage_risk,
        fixes=fixes,
        drift_metrics=drift_metrics,
        calibration_details=calibration_details,
        shortcut_analysis=shortcut_analysis,
        saturation_analysis=saturation_analysis,
        length_bias_detected=length_bias_detected,
        length_bias_metrics=length_bias_metrics,
        length_bias_recommendations=length_bias_recommendations,
    )


def _detect_saturation(
    reward_values: pd.Series, threshold: float
) -> Tuple[List[str], Dict[str, Any]]:
    """Detect reward saturation at boundaries."""
    issues = []
    analysis = {}

    # Check for clustering at boundaries
    total_samples = len(reward_values)

    # Check upper bound (assuming rewards are normalized to [0,1] or [-1,1])
    upper_threshold = 0.95
    lower_threshold = -0.95

    upper_saturated = (reward_values >= upper_threshold).sum()
    lower_saturated = (reward_values <= lower_threshold).sum()

    upper_ratio = upper_saturated / total_samples
    lower_ratio = lower_saturated / total_samples

    analysis["upper_saturation_ratio"] = upper_ratio
    analysis["lower_saturation_ratio"] = lower_ratio
    analysis["total_samples"] = total_samples

    if upper_ratio > threshold:
        issues.append(
            f"High upper saturation: {upper_ratio:.1%} of rewards at upper bound"
        )

    if lower_ratio > threshold:
        issues.append(
            f"High lower saturation: {lower_ratio:.1%} of rewards at lower bound"
        )

    # Check for zero clustering (common in RLHF)
    zero_threshold = 0.05
    zero_rewards = (np.abs(reward_values) <= zero_threshold).sum()
    zero_ratio = zero_rewards / total_samples

    analysis["zero_ratio"] = zero_ratio

    if zero_ratio > threshold:
        issues.append(f"High zero clustering: {zero_ratio:.1%} of rewards near zero")

    return issues, analysis


def _detect_shortcut_signals(
    run_data: pd.DataFrame, reward_col: str, threshold: float
) -> Tuple[List[str], Dict[str, float]]:
    """Detect shortcut signals that reward model might be exploiting."""
    issues = []
    analysis = {}

    # Length bias detection is handled by LengthBiasDetector in ``health``.
    # Avoid duplicating alerts here so we can control severity thresholds in one place.

    # Check for repetition bias
    if "repetition_penalty" in run_data.columns:
        try:
            rep_corr = np.corrcoef(run_data["repetition_penalty"], run_data[reward_col])[
                0, 1
            ]
            if not np.isnan(rep_corr):
                analysis["repetition_correlation"] = float(rep_corr)
                if abs(rep_corr) > threshold:
                    issues.append(f"Repetition bias detected: correlation {rep_corr:.3f}")
        except (ValueError, np.linalg.LinAlgError):
            pass

    # Check for formatting bias (e.g., markdown, code blocks)
    if "has_markdown" in run_data.columns:
        try:
            markdown_corr = np.corrcoef(run_data["has_markdown"], run_data[reward_col])[
                0, 1
            ]
            if not np.isnan(markdown_corr):
                analysis["markdown_correlation"] = float(markdown_corr)
                if abs(markdown_corr) > threshold:
                    issues.append(f"Markdown bias detected: correlation {markdown_corr:.3f}")
        except (ValueError, np.linalg.LinAlgError):
            pass

    # Check for keyword bias
    if "keyword_count" in run_data.columns:
        try:
            keyword_corr = np.corrcoef(run_data["keyword_count"], run_data[reward_col])[
                0, 1
            ]
            if not np.isnan(keyword_corr):
                analysis["keyword_correlation"] = float(keyword_corr)
                if abs(keyword_corr) > threshold:
                    issues.append(f"Keyword bias detected: correlation {keyword_corr:.3f}")
        except (ValueError, np.linalg.LinAlgError):
            pass

    return issues, analysis


def _prepare_length_bias_inputs(
    run_data: pd.DataFrame,
    response_col: Optional[str],
    length_col: Optional[str],
) -> Tuple[Optional[List[Any]], Optional[List[Optional[float]]]]:
    """Collect response text and length information for length-bias analysis."""

    responses: Optional[List[Any]] = None
    lengths: Optional[List[Optional[float]]] = None

    candidate_response_cols: Tuple[Optional[str], ...] = (
        response_col,
        "response_text",
        "response",
        "completion",
        "output_text",
    )
    for column in candidate_response_cols:
        if column and column in run_data.columns:
            responses = run_data[column].tolist()
            break

    candidate_length_cols: Tuple[Optional[str], ...] = (
        length_col,
        "response_length",
        "response_tokens",
        "tokens_out",
        "token_count",
        "length",
    )
    for column in candidate_length_cols:
        if column and column in run_data.columns:
            raw_values = run_data[column].tolist()
            converted: List[Optional[float]] = []
            for value in raw_values:
                if value is None:
                    converted.append(None)
                else:
                    try:
                        converted.append(float(value))
                    except (TypeError, ValueError):
                        converted.append(None)
            lengths = converted
            break

    if responses is not None and len(responses) == 0:
        responses = None

    if lengths is not None and len(lengths) == 0:
        lengths = None

    if responses is not None and lengths is None:
        inferred_lengths: List[Optional[float]] = []
        has_valid_length = False
        for value in responses:
            if value is None:
                inferred_lengths.append(None)
                continue
            try:
                length_value = float(len(str(value)))
            except Exception:
                inferred_lengths.append(None)
                continue
            inferred_lengths.append(length_value)
            has_valid_length = True
        lengths = inferred_lengths if has_valid_length else None

    if responses is None and lengths is not None:
        # Ensure the detector receives a placeholder iterable with the correct length.
        responses = [""] * len(lengths)

    if responses is not None and lengths is not None:
        if len(responses) != len(lengths):
            raise ValueError(
                "Response and length columns must have matching sample counts"
            )

        if all(length is None for length in lengths):
            lengths = None

    if (responses is None) != (lengths is None):
        # If lengths could not be inferred, skip detection to avoid mismatched inputs.
        responses = None
        lengths = None

    return responses, lengths


def _detect_label_leakage(
    run_data: pd.DataFrame, reward_col: str, threshold: float
) -> float:
    """Detect potential label leakage in reward model."""
    leakage_risk = 0.0

    # Check if reward model has access to training metadata it shouldn't
    suspicious_cols = ["epoch", "batch_idx", "run_id", "git_sha", "timestamp"]

    for col in suspicious_cols:
        if col in run_data.columns:
            # Calculate correlation with rewards
            try:
                corr = np.corrcoef(run_data[col].astype(float), run_data[reward_col])[
                    0, 1
                ]
                if not np.isnan(corr) and abs(corr) > threshold:
                    leakage_risk += (
                        0.3  # Increment risk for each suspicious correlation
                    )
            except (ValueError, TypeError):
                # Column might not be numeric, skip
                continue

    # Check for perfect correlation with step (indicates overfitting)
    if "step" in run_data.columns:
        step_corr = np.corrcoef(run_data["step"], run_data[reward_col])[0, 1]
        if abs(step_corr) > 0.8:
            leakage_risk += 0.4

    return min(leakage_risk, 1.0)
