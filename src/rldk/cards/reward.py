"""Reward card generation for RL training runs."""

import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from ..io.event_schema import Event


def generate_reward_card(
    events: List[Event], run_path: str, output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a reward health card for a training run.

    Args:
        events: List of Event objects from the training run
        run_path: Path to the training run
        output_dir: Directory to save the card (defaults to runs/run_id/rldk_cards)

    Returns:
        Dictionary containing the reward card data
    """
    # Extract run_id from events
    run_id = events[0].model_info["run_id"] if events else "unknown"

    # Set output directory
    if output_dir is None:
        output_dir = f"runs/{run_id}/rldk_cards"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Analyze reward health
    reward_analysis = _analyze_reward_health(events)

    # Detect drift patterns
    drift_detected = _detect_reward_drift(events)

    # Calculate calibration score
    calibration_score = _calculate_calibration_score(events)

    # Detect saturation
    saturation_detected = _detect_reward_saturation(events)

    # Find shortcut signals
    shortcut_signals = _detect_shortcut_signals(events)

    # Estimate label noise
    label_noise = _estimate_label_noise(events)

    # Perform slice analysis
    slice_analysis = _perform_slice_analysis(events)

    # Generate recommendations
    recommendations = _generate_reward_recommendations(
        reward_analysis,
        drift_detected,
        calibration_score,
        saturation_detected,
        shortcut_signals,
        label_noise,
    )

    # Create the card data
    card_data = {
        "version": "1.0",
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(),
        "passed": bool(
            _evaluate_reward_health(reward_analysis, drift_detected, calibration_score)
        ),
        "drift_detected": bool(drift_detected),
        "calibration_score": float(calibration_score),
        "saturation_detected": bool(saturation_detected),
        "shortcut_signals": shortcut_signals,
        "label_noise": float(label_noise),
        "metrics": {
            "correlation": float(reward_analysis.get("correlation", 0)),
            "mae": float(reward_analysis.get("mae", 0)),
            "l2_distance": float(reward_analysis.get("l2_distance", 0)),
        },
        "slice_analysis": slice_analysis,
        "recommendations": recommendations,
    }

    # Save JSON card
    json_path = output_path / "reward_card.json"
    with open(json_path, "w") as f:
        # Convert numpy types to native Python types for JSON serialization
        json.dump(
            card_data,
            f,
            indent=2,
            default=lambda x: float(x) if hasattr(x, "item") else x,
        )

    # Generate and save PNG visualization
    png_path = output_path / "reward_card.png"
    _generate_reward_visualization(card_data, png_path)

    return card_data


def _analyze_reward_health(events: List[Event]) -> Dict[str, float]:
    """Analyze reward health metrics."""
    if not events:
        return {}

    # Convert events to DataFrame
    from ..io.event_schema import events_to_dataframe

    df = events_to_dataframe(events)

    # Calculate basic statistics
    analysis = {}

    if "reward_mean" in df.columns:
        rewards = df["reward_mean"].dropna()
        if len(rewards) > 0:
            analysis["mean"] = float(rewards.mean())
            analysis["std"] = float(rewards.std())
            analysis["min"] = float(rewards.min())
            analysis["max"] = float(rewards.max())

            # Calculate trend
            if len(rewards) > 1:
                x = np.arange(len(rewards))
                slope = np.polyfit(x, rewards, 1)[0]
                analysis["trend"] = float(slope)

    # Calculate correlation with other metrics
    if "reward_mean" in df.columns and "kl_mean" in df.columns:
        corr = df["reward_mean"].corr(df["kl_mean"])
        analysis["correlation"] = float(corr) if not pd.isna(corr) else 0.0

    # Calculate MAE and L2 distance (assuming some baseline)
    if "reward_mean" in df.columns:
        rewards = df["reward_mean"].dropna()
        if len(rewards) > 0:
            # Use first value as baseline for comparison
            baseline = rewards.iloc[0]
            mae = np.mean(np.abs(rewards - baseline))
            l2_distance = np.sqrt(np.mean((rewards - baseline) ** 2))
            analysis["mae"] = float(mae)
            analysis["l2_distance"] = float(l2_distance)

    return analysis


def _detect_reward_drift(events: List[Event]) -> bool:
    """Detect if there's significant reward drift."""
    if not events:
        return False

    # Convert events to DataFrame
    from ..io.event_schema import events_to_dataframe

    df = events_to_dataframe(events)

    if "reward_mean" not in df.columns:
        return False

    rewards = df["reward_mean"].dropna()
    if len(rewards) < 10:
        return False

    # Split into early and late periods
    mid_point = len(rewards) // 2
    early_rewards = rewards[:mid_point]
    late_rewards = rewards[mid_point:]

    if len(early_rewards) == 0 or len(late_rewards) == 0:
        return False

    # Calculate drift
    early_mean = early_rewards.mean()
    late_mean = late_rewards.mean()
    drift_magnitude = abs(late_mean - early_mean)

    # Consider drift if change is more than 20% of early mean
    return drift_magnitude > 0.2 * abs(early_mean)


def _calculate_calibration_score(events: List[Event]) -> float:
    """Calculate calibration score for reward model."""
    if not events:
        return 0.0

    # Convert events to DataFrame
    from ..io.event_schema import events_to_dataframe

    df = events_to_dataframe(events)

    if "reward_mean" not in df.columns or "reward_std" not in df.columns:
        return 0.0

    rewards = df["reward_mean"].dropna()
    reward_stds = df["reward_std"].dropna()

    if len(rewards) == 0 or len(reward_stds) == 0:
        return 0.0

    # Calculate calibration based on consistency between mean and std
    # Higher score means more consistent calibration
    calibration_scores = []

    # Calculate expected standard deviation based on reward distribution
    reward_mean = np.mean(rewards)
    reward_range = np.max(rewards) - np.min(rewards)
    
    # For well-calibrated models, std should be proportional to the reward range
    # Use empirical relationship: ideal_std ≈ 0.1 * reward_range for normalized rewards
    # or ideal_std ≈ 0.1 * abs(reward_mean) for centered rewards
    if reward_range > 0:
        # For rewards with significant range, use range-based estimation
        ideal_std = min(0.3, reward_range * 0.2)  # Conservative estimate
    elif abs(reward_mean) > 0:
        # For centered rewards, use mean-based estimation
        ideal_std = min(0.3, abs(reward_mean) * 0.1)
    else:
        # Fallback for edge cases
        ideal_std = 0.1

    for i in range(len(rewards)):
        if i < len(reward_stds):
            mean_val = rewards.iloc[i]
            std_val = reward_stds.iloc[i]

            if std_val > 0:
                # Score based on how well std reflects the expected variability
                # Use relative error instead of absolute difference
                relative_error = abs(std_val - ideal_std) / max(ideal_std, 0.01)
                score = 1.0 / (1.0 + relative_error)
                calibration_scores.append(score)

    if calibration_scores:
        return float(np.mean(calibration_scores))

    return 0.0


def _detect_reward_saturation(events: List[Event]) -> bool:
    """Detect if reward model is saturated."""
    if not events:
        return False

    # Convert events to DataFrame
    from ..io.event_schema import events_to_dataframe

    df = events_to_dataframe(events)

    if "reward_mean" not in df.columns:
        return False

    rewards = df["reward_mean"].dropna()
    if len(rewards) < 10:
        return False

    # Check for saturation (rewards staying very high or very low)
    recent_rewards = rewards.tail(10)  # Last 10 steps

    # Check if rewards are consistently high (>0.8) or low (<0.2)
    high_saturation = (recent_rewards > 0.8).mean() > 0.7
    low_saturation = (recent_rewards < 0.2).mean() > 0.7

    return high_saturation or low_saturation


def _detect_shortcut_signals(events: List[Event]) -> List[str]:
    """Detect potential shortcut learning signals."""
    signals = []

    if not events:
        return signals

    # Convert events to DataFrame
    from ..io.event_schema import events_to_dataframe

    df = events_to_dataframe(events)

    # Check for reward hacking patterns
    if "reward_mean" in df.columns and "kl_mean" in df.columns:
        rewards = df["reward_mean"].dropna()
        kl_divs = df["kl_mean"].dropna()

        if len(rewards) > 0 and len(kl_divs) > 0:
            # Check for negative correlation (reward up, KL down)
            corr = rewards.corr(kl_divs)
            if corr < -0.5:
                signals.append("Negative reward-KL correlation detected")

    # Check for gradient norm issues
    if "grad_norm" in df.columns:
        grad_norms = df["grad_norm"].dropna()
        if len(grad_norms) > 0:
            if grad_norms.max() > 10.0:
                signals.append("Large gradient norms detected")
            if grad_norms.std() < 0.1:
                signals.append("Very stable gradient norms (potential shortcut)")

    # Check for entropy collapse
    if "entropy_mean" in df.columns:
        entropies = df["entropy_mean"].dropna()
        if len(entropies) > 0:
            if entropies.min() < 0.1:
                signals.append("Low entropy detected (potential mode collapse)")

    return signals


def _estimate_label_noise(events: List[Event]) -> float:
    """Estimate label noise in the reward model."""
    if not events:
        return 0.0

    # Convert events to DataFrame
    from ..io.event_schema import events_to_dataframe

    df = events_to_dataframe(events)

    if "reward_std" not in df.columns:
        return 0.0

    reward_stds = df["reward_std"].dropna()
    if len(reward_stds) == 0:
        return 0.0

    # Estimate noise as average standard deviation
    noise_estimate = float(reward_stds.mean())

    # Normalize to 0-1 range
    return min(noise_estimate, 1.0)


def _perform_slice_analysis(events: List[Event]) -> Dict[str, Dict[str, Any]]:
    """Perform analysis on different data slices."""
    slice_analysis = {}

    if not events:
        return slice_analysis

    # Convert events to DataFrame
    from ..io.event_schema import events_to_dataframe

    df = events_to_dataframe(events)

    if "reward_mean" not in df.columns:
        return slice_analysis

    # Analyze different phases
    if "phase" in df.columns:
        for phase in df["phase"].unique():
            if pd.notna(phase):
                phase_data = df[df["phase"] == phase]["reward_mean"].dropna()
                if len(phase_data) > 0:
                    slice_analysis[phase] = {
                        "delta_mean": float(phase_data.mean()),
                        "n_samples": int(len(phase_data)),
                    }

    # Analyze by step ranges
    if len(df) > 20:
        early_data = df.head(10)["reward_mean"].dropna()
        late_data = df.tail(10)["reward_mean"].dropna()

        if len(early_data) > 0 and len(late_data) > 0:
            slice_analysis["early_vs_late"] = {
                "delta_mean": float(late_data.mean() - early_data.mean()),
                "n_samples": int(len(early_data) + len(late_data)),
            }

    # Add some default slices if none exist
    if not slice_analysis:
        rewards = df["reward_mean"].dropna()
        if len(rewards) > 0:
            slice_analysis["overall"] = {
                "delta_mean": float(rewards.mean()),
                "n_samples": int(len(rewards)),
            }

    return slice_analysis


def _generate_reward_recommendations(
    reward_analysis: Dict[str, float],
    drift_detected: bool,
    calibration_score: float,
    saturation_detected: bool,
    shortcut_signals: List[str],
    label_noise: float,
) -> List[str]:
    """Generate recommendations based on reward analysis."""
    recommendations = []

    # Calibration recommendations
    if calibration_score < 0.7:
        recommendations.append("Consider recalibrating reward model")

    # Drift recommendations
    if drift_detected:
        recommendations.append("Monitor reward drift over time")
        recommendations.append("Consider reward model retraining")

    # Saturation recommendations
    if saturation_detected:
        recommendations.append("Reward model may be saturated - check reward scale")
        recommendations.append("Consider adjusting reward function")

    # Shortcut learning recommendations
    if shortcut_signals:
        recommendations.append("Investigate potential shortcut learning")
        recommendations.append("Review reward function design")

    # Label noise recommendations
    if label_noise > 0.1:
        recommendations.append("High label noise detected - improve data quality")

    # General recommendations
    if reward_analysis.get("correlation", 0) < 0.8:
        recommendations.append("Consider recalibration if correlation drops below 0.8")

    if not recommendations:
        recommendations.append("Reward model appears healthy")
        recommendations.append("Continue monitoring for drift")

    return recommendations


def _evaluate_reward_health(
    reward_analysis: Dict[str, float], drift_detected: bool, calibration_score: float
) -> bool:
    """Evaluate overall reward health."""
    # Consider healthy if:
    # - No significant drift
    # - Good calibration score
    # - Reasonable correlation

    if drift_detected:
        return False

    if calibration_score < 0.6:
        return False

    if reward_analysis.get("correlation", 0) < 0.7:
        return False

    return True


def _generate_reward_visualization(
    card_data: Dict[str, Any], output_path: Path
) -> None:
    """Generate a visual representation of the reward card."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Reward Card - {card_data['run_id']}", fontsize=16, fontweight="bold")

    # Overall status
    status_color = "green" if card_data["passed"] else "red"
    status_text = "HEALTHY" if card_data["passed"] else "ISSUES"

    ax1.text(
        0.5,
        0.5,
        status_text,
        fontsize=36,
        ha="center",
        va="center",
        color=status_color,
        fontweight="bold",
    )
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")
    ax1.set_title("Reward Health Status", fontsize=14, fontweight="bold")

    # Key metrics
    metrics = card_data["metrics"]
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    bars = ax2.bar(
        metric_names,
        metric_values,
        color=[
            "green" if v > 0.8 else "orange" if v > 0.5 else "red"
            for v in metric_values
        ],
    )
    ax2.set_title("Key Metrics", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Score")
    ax2.tick_params(axis="x", rotation=45)

    # Add threshold lines
    ax2.axhline(y=0.8, color="green", linestyle="--", alpha=0.7, label="Good")
    ax2.axhline(y=0.5, color="orange", linestyle="--", alpha=0.7, label="Fair")
    ax2.legend()

    # Calibration and drift
    calib_score = card_data["calibration_score"]
    drift_detected = card_data["drift_detected"]

    ax3.text(
        0.5,
        0.7,
        f"Calibration Score:\n{calib_score:.2f}",
        fontsize=16,
        ha="center",
        va="center",
        fontweight="bold",
    )
    ax3.text(
        0.5,
        0.3,
        f"Drift Detected:\n{'Yes' if drift_detected else 'No'}",
        fontsize=16,
        ha="center",
        va="center",
        color="red" if drift_detected else "green",
    )
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis("off")
    ax3.set_title("Calibration & Drift", fontsize=14, fontweight="bold")

    # Issues and recommendations
    issues = []
    if card_data["saturation_detected"]:
        issues.append("Saturation detected")
    if card_data["shortcut_signals"]:
        issues.append(f"{len(card_data['shortcut_signals'])} shortcut signals")
    if card_data["label_noise"] > 0.1:
        issues.append("High label noise")

    issues_text = (
        "\n".join([f"• {issue}" for issue in issues])
        if issues
        else "No issues detected"
    )
    recs = card_data["recommendations"][:3]  # Show first 3 recommendations
    recs_text = "\n".join([f"• {rec}" for rec in recs])

    ax4.text(
        0.05,
        0.95,
        "Issues:",
        fontsize=12,
        fontweight="bold",
        va="top",
        transform=ax4.transAxes,
    )
    ax4.text(0.05, 0.85, issues_text, fontsize=9, va="top", transform=ax4.transAxes)
    ax4.text(
        0.05,
        0.45,
        "Recommendations:",
        fontsize=12,
        fontweight="bold",
        va="top",
        transform=ax4.transAxes,
    )
    ax4.text(0.05, 0.35, recs_text, fontsize=9, va="top", transform=ax4.transAxes)

    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis("off")
    ax4.set_title("Issues & Recommendations", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
