"""Main reward health checking functionality."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import contextlib
import io

import numpy as np
import pandas as pd

from .calibration import analyze_calibration
from ..forensics.kl_schedule_tracker import KLScheduleTracker
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
    overoptimization: "OveroptimizationAnalysis" = field(
        default_factory=lambda: OveroptimizationAnalysis()
    )


@dataclass
class OveroptimizationAnalysis:
    """Summary of reward-vs-gold overoptimization analysis."""

    proxy_improvement: float = 0.0
    gold_improvement: float = 0.0
    delta: float = 0.0
    correlation_trend: Dict[str, Optional[float]] = field(default_factory=dict)
    kl_summary: Dict[str, Any] = field(default_factory=dict)
    flagged: bool = False
    gold_metrics_available: bool = False
    gold_regressed: bool = False
    gold_stagnant: bool = False
    kl_elevated: bool = False
    correlation_declined: bool = False
    warning: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    window_size: int = 100
    delta_threshold: float = 0.2
    min_samples: int = 100
    sample_size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""

        def _normalize_value(value: Any) -> Any:
            if isinstance(value, bool):
                return value
            if isinstance(value, (np.floating, np.integer)):
                return float(value)
            if isinstance(value, (float, int)):
                if pd.isna(value):
                    return None
                return float(value)
            if isinstance(value, dict):
                return {k: _normalize_value(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_normalize_value(v) for v in value]
            if pd.isna(value):
                return None
            return value

        payload = asdict(self)
        return {key: _normalize_value(val) for key, val in payload.items()}


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
    gold_metrics: Optional[Union[pd.DataFrame, pd.Series]] = None,
    gold_metric_col: Optional[str] = None,
    overoptimization_window: int = 100,
    overoptimization_delta_threshold: float = 0.2,
    overoptimization_min_samples: int = 100,
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
        gold_metrics: Optional trusted gold metric timeseries
        gold_metric_col: Column name to pull gold metrics from if available
        overoptimization_window: Window size for early/late comparisons
        overoptimization_delta_threshold: Proxy-minus-gold delta to flag issues
        overoptimization_min_samples: Minimum paired samples required for detector

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

    overoptimization_analysis = _analyze_overoptimization(
        run_data=run_data,
        reward_col=reward_col,
        step_col=step_col,
        gold_metrics=gold_metrics,
        gold_metric_col=gold_metric_col,
        window_size=overoptimization_window,
        delta_threshold=overoptimization_delta_threshold,
        min_samples=overoptimization_min_samples,
    )

    if overoptimization_analysis.warning:
        warning_message = overoptimization_analysis.warning
        if warning_message not in fixes:
            fixes.append(warning_message)
    elif overoptimization_analysis.flagged:
        issues.append(
            "Reward overoptimization suspected: proxy reward improved while gold metrics stagnated"
        )
        fixes.append(
            "Pause reward optimization, review KL controls, and refresh gold evaluations to realign the reward model"
        )

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
        overoptimization=overoptimization_analysis,
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


def _analyze_overoptimization(
    run_data: pd.DataFrame,
    reward_col: str,
    step_col: str,
    gold_metrics: Optional[Union[pd.DataFrame, pd.Series]],
    gold_metric_col: Optional[str],
    window_size: int,
    delta_threshold: float,
    min_samples: int,
) -> OveroptimizationAnalysis:
    analysis = OveroptimizationAnalysis(
        window_size=window_size,
        delta_threshold=delta_threshold,
        min_samples=min_samples,
    )

    gold_df, warning = _prepare_gold_dataframe(
        run_data=run_data,
        step_col=step_col,
        gold_metrics=gold_metrics,
        gold_metric_col=gold_metric_col,
    )

    if gold_df is None:
        if warning:
            analysis.warning = warning
        return analysis

    analysis.gold_metrics_available = True

    paired = run_data[[step_col, reward_col]].merge(gold_df, on=step_col, how="inner")
    paired = paired.dropna(subset=[reward_col, "gold_metric"])
    paired = paired.sort_values(step_col).reset_index(drop=True)
    sample_size = len(paired)
    analysis.sample_size = sample_size

    if sample_size < max(4, min_samples):
        analysis.warning = (
            "Insufficient overlapping reward/gold samples for overoptimization analysis"
        )
        return analysis

    effective_window = max(2, min(window_size, sample_size // 2))
    if effective_window < 2:
        analysis.warning = "Not enough data points to compute early/late window statistics"
        return analysis

    early_window = paired.iloc[:effective_window]
    late_window = paired.iloc[-effective_window:]

    proxy_improvement = late_window[reward_col].mean() - early_window[reward_col].mean()
    gold_improvement = late_window["gold_metric"].mean() - early_window["gold_metric"].mean()
    analysis.proxy_improvement = float(proxy_improvement)
    analysis.gold_improvement = float(gold_improvement)
    analysis.delta = float(proxy_improvement - gold_improvement)
    analysis.gold_regressed = gold_improvement < 0
    analysis.gold_stagnant = abs(gold_improvement) < delta_threshold * 0.25

    correlation_trend = {}
    for method in ("pearson", "spearman"):
        early_corr = _safe_correlation(early_window[reward_col], early_window["gold_metric"], method)
        late_corr = _safe_correlation(late_window[reward_col], late_window["gold_metric"], method)
        if early_corr is not None:
            correlation_trend[f"{method}_early"] = early_corr
        if late_corr is not None:
            correlation_trend[f"{method}_late"] = late_corr
        if early_corr is not None and late_corr is not None:
            correlation_trend[f"{method}_delta"] = float(late_corr - early_corr)

    analysis.correlation_trend = correlation_trend
    deltas = [correlation_trend.get("pearson_delta"), correlation_trend.get("spearman_delta")]
    analysis.correlation_declined = any(
        delta is not None and delta < -0.05 for delta in deltas if delta is not None
    )

    kl_summary = _summarize_recent_kl(run_data, step_col)
    analysis.kl_summary = kl_summary
    current_mean = _safe_numeric(kl_summary.get("kl_current_mean"))
    if current_mean is None:
        current_mean = _safe_numeric(kl_summary.get("current_kl"))
    kl_target = _safe_numeric(kl_summary.get("kl_target")) or 0.1
    analysis.kl_elevated = (
        current_mean is not None and current_mean > kl_target * 1.5
    ) or bool(kl_summary.get("target_range_violations", 0))

    proxy_gain_ok = proxy_improvement >= delta_threshold
    gold_flat = analysis.gold_stagnant or analysis.gold_regressed or gold_improvement <= 0

    if (
        proxy_gain_ok
        and gold_flat
        and analysis.kl_elevated
        and sample_size >= min_samples
    ):
        analysis.flagged = True
        if not analysis.correlation_declined:
            analysis.notes.append(
                "Proxy/gold delta cleared the threshold but correlations remained stable"
            )

    return analysis


def _prepare_gold_dataframe(
    run_data: pd.DataFrame,
    step_col: str,
    gold_metrics: Optional[Union[pd.DataFrame, pd.Series]],
    gold_metric_col: Optional[str],
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    candidate: Optional[pd.DataFrame]

    if gold_metrics is None:
        candidate_cols = [
            gold_metric_col,
            "gold_metric",
            "gold_score",
            "trusted_score",
            "eval_score",
            "benchmark_score",
        ]
        resolved_col = next(
            (col for col in candidate_cols if col and col in run_data.columns),
            None,
        )
        if resolved_col is None:
            return None, "Gold metrics not provided; supply --gold-col or --gold path to enable overoptimization checks"
        candidate = run_data[[step_col, resolved_col]].rename(
            columns={resolved_col: "gold_metric"}
        )
        return candidate, None

    if isinstance(gold_metrics, pd.Series):
        series = gold_metrics.dropna()
        if series.empty:
            return None, "Provided gold metrics are empty"
        if series.index.name == step_col or series.index.equals(run_data[step_col]):
            df = series.to_frame(name="gold_metric").reset_index()
            if series.index.name != step_col:
                df = df.rename(columns={"index": step_col})
            return df[[step_col, "gold_metric"]], None
        if len(series) == len(run_data):
            df = pd.DataFrame({step_col: run_data[step_col], "gold_metric": series.values})
            return df, None
        return None, "Gold metrics series length does not match reward metrics"

    if isinstance(gold_metrics, pd.DataFrame):
        gold_df = gold_metrics.copy()
        if step_col not in gold_df.columns:
            if len(gold_df) == len(run_data):
                gold_df = gold_df.copy()
                gold_df[step_col] = run_data[step_col].values
            else:
                return None, "Gold metrics missing step information"

        candidate_cols = [
            gold_metric_col,
            "gold_metric",
            "gold_score",
            "trusted_score",
            "eval_score",
            "benchmark_score",
        ]
        resolved_col = next(
            (col for col in candidate_cols if col and col in gold_df.columns),
            None,
        )
        if resolved_col is None:
            remaining = [c for c in gold_df.columns if c != step_col]
            if len(remaining) == 1:
                resolved_col = remaining[0]
        if resolved_col is None:
            return None, "Unable to determine gold metric column"
        return (
            gold_df[[step_col, resolved_col]].rename(columns={resolved_col: "gold_metric"}),
            None,
        )

    return None, "Unsupported gold metrics format"


def _safe_correlation(
    rewards: pd.Series, gold: pd.Series, method: str
) -> Optional[float]:
    if rewards.nunique(dropna=True) < 2 or gold.nunique(dropna=True) < 2:
        return None
    try:
        corr = rewards.corr(gold, method=method)
        if pd.isna(corr):
            return None
        return float(corr)
    except Exception:
        return None


def _summarize_recent_kl(run_data: pd.DataFrame, step_col: str) -> Dict[str, Any]:
    candidate_cols = ["kl_mean", "kl", "kl_value", "kl_divergence"]
    kl_col = next((col for col in candidate_cols if col in run_data.columns), None)
    if kl_col is None:
        return {}

    coef_cols = ["kl_coef", "kl_coefficient", "kl_beta", "kl_scale"]
    coef_col = next((col for col in coef_cols if col in run_data.columns), None)

    with contextlib.redirect_stdout(io.StringIO()):
        tracker = KLScheduleTracker(enable_drift_tracking=False)

    for step, row in run_data[[step_col, kl_col]].dropna().iterrows():
        step_value = int(row[step_col]) if step_col in run_data.columns else int(step)
        kl_value = row[kl_col]
        coef_value = 1.0
        if coef_col is not None:
            coef_value = run_data.loc[row.name, coef_col]
        tracker.update(step_value, kl_value, coef_value)

    summary = tracker.get_summary()
    sanitized = {key: _safe_numeric(value) if isinstance(value, (float, int, np.floating, np.integer)) else value for key, value in summary.items()}
    return sanitized


def _safe_numeric(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric
