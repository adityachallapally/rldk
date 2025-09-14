"""Plotting utilities for reward health analysis."""

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def generate_calibration_plots(
    reward_data: pd.DataFrame,
    reward_col: str,
    human_pref_col: Optional[str] = None,
    ground_truth_col: Optional[str] = None,
    output_path: Optional[Path] = None,
    figsize: tuple = (12, 8),
) -> Dict[str, Any]:
    """
    Generate calibration plots for reward model analysis.

    Args:
        reward_data: Training run data
        reward_col: Column name for reward values
        human_pref_col: Column name for human preference scores
        ground_truth_col: Column name for ground truth labels
        output_path: Path to save plots
        figsize: Figure size for plots

    Returns:
        Dictionary with plot information and saved paths
    """

    plots_info = {}

    # Set style
    plt.style.use("default")
    sns.set_palette("husl")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Reward Model Calibration Analysis", fontsize=16, fontweight="bold")

    # 1. Reward distribution over time
    if "step" in reward_data.columns:
        ax1 = axes[0, 0]
        _plot_reward_timeline(ax1, reward_data, reward_col)
        plots_info["timeline_plot"] = "Reward timeline plot generated"

    # 2. Reward distribution histogram
    ax2 = axes[0, 1]
    _plot_reward_distribution(ax2, reward_data[reward_col])
    plots_info["distribution_plot"] = "Reward distribution plot generated"

    # 3. Calibration plot (if human preference data available)
    if human_pref_col and human_pref_col in reward_data.columns:
        ax3 = axes[1, 0]
        _plot_calibration_curve(ax3, reward_data, reward_col, human_pref_col)
        plots_info["calibration_plot"] = "Calibration curve plot generated"

    # 4. Correlation analysis
    ax4 = axes[1, 1]
    if human_pref_col and human_pref_col in reward_data.columns:
        _plot_correlation(
            ax4, reward_data, reward_col, human_pref_col, "Human Preference"
        )
    elif ground_truth_col and ground_truth_col in reward_data.columns:
        _plot_correlation(
            ax4, reward_data, reward_col, ground_truth_col, "Ground Truth"
        )
    else:
        _plot_reward_statistics(ax4, reward_data, reward_col)

    plots_info["correlation_plot"] = "Correlation/statistics plot generated"

    # Adjust layout
    plt.tight_layout()

    # Save plot if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plots_info["saved_path"] = str(output_path)

    return plots_info


def _plot_reward_timeline(ax: plt.Axes, data: pd.DataFrame, reward_col: str):
    """Plot reward values over training steps."""

    if "step" not in data.columns:
        ax.text(
            0.5,
            0.5,
            "No step data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Reward Timeline")
        return

    # Sample data if too many points
    if len(data) > 1000:
        sample_data = data.sample(n=1000, random_state=42).sort_values("step")
    else:
        sample_data = data.sort_values("step")

    ax.plot(sample_data["step"], sample_data[reward_col], alpha=0.7, linewidth=1)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Reward Value")
    ax.set_title("Reward Timeline")
    ax.grid(True, alpha=0.3)

    # Add trend line
    if len(sample_data) > 10:
        z = np.polyfit(sample_data["step"], sample_data[reward_col], 1)
        p = np.poly1d(z)
        ax.plot(
            sample_data["step"], p(sample_data["step"]), "r--", alpha=0.8, linewidth=2
        )


def _plot_reward_distribution(ax: plt.Axes, reward_values: pd.Series):
    """Plot reward value distribution histogram."""

    # Remove NaN values
    clean_values = reward_values.dropna()

    if len(clean_values) == 0:
        ax.text(
            0.5,
            0.5,
            "No valid reward data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Reward Distribution")
        return

    # Create histogram
    ax.hist(clean_values, bins=30, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Reward Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Reward Distribution")
    ax.grid(True, alpha=0.3)

    # Add statistics
    mean_val = clean_values.mean()
    std_val = clean_values.std()
    ax.axvline(
        mean_val, color="red", linestyle="--", alpha=0.8, label=f"Mean: {mean_val:.3f}"
    )
    ax.axvline(
        mean_val + std_val,
        color="orange",
        linestyle=":",
        alpha=0.6,
        label=f"+1σ: {mean_val + std_val:.3f}",
    )
    ax.axvline(
        mean_val - std_val,
        color="orange",
        linestyle=":",
        alpha=0.6,
        label=f"-1σ: {mean_val - std_val:.3f}",
    )
    ax.legend()


def _plot_calibration_curve(
    ax: plt.Axes, data: pd.DataFrame, reward_col: str, human_pref_col: str
):
    """Plot calibration curve for reward vs human preference."""

    # Remove NaN values
    valid_mask = ~(data[reward_col].isna() | data[human_pref_col].isna())
    valid_data = data[valid_mask]

    if len(valid_data) < 10:
        ax.text(
            0.5,
            0.5,
            "Insufficient data for calibration",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Calibration Curve")
        return

    # Normalize reward scores to [0, 1]
    reward_scores = valid_data[reward_col].values
    human_scores = valid_data[human_pref_col].values

    reward_normalized = (reward_scores - reward_scores.min()) / (
        reward_scores.max() - reward_scores.min()
    )

    # Create calibration curve
    try:
        from sklearn.calibration import calibration_curve

        fraction_of_positives, mean_predicted_value = calibration_curve(
            human_scores, reward_normalized, n_bins=10
        )

        # Plot calibration curve
        ax.plot(
            mean_predicted_value, fraction_of_positives, "o-", label="Calibration Curve"
        )
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title("Calibration Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add correlation info
        corr = np.corrcoef(reward_scores, human_scores)[0, 1]
        ax.text(
            0.05,
            0.95,
            f"Correlation: {corr:.3f}",
            transform=ax.transAxes,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    except Exception as e:
        ax.text(
            0.5,
            0.5,
            f"Calibration failed: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Calibration Curve")


def _plot_correlation(
    ax: plt.Axes, data: pd.DataFrame, reward_col: str, other_col: str, other_name: str
):
    """Plot correlation between reward and another metric."""

    # Remove NaN values
    valid_mask = ~(data[reward_col].isna() | data[other_col].isna())
    valid_data = data[valid_mask]

    if len(valid_data) < 10:
        ax.text(
            0.5,
            0.5,
            "Insufficient data for correlation",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(f"Reward vs {other_name}")
        return

    # Create scatter plot
    ax.scatter(valid_data[reward_col], valid_data[other_col], alpha=0.6, s=20)

    # Add trend line
    z = np.polyfit(valid_data[reward_col], valid_data[other_col], 1)
    p = np.poly1d(z)
    ax.plot(
        valid_data[reward_col], p(valid_data[reward_col]), "r--", alpha=0.8, linewidth=2
    )

    ax.set_xlabel("Reward Value")
    ax.set_ylabel(other_name)
    ax.set_title(f"Reward vs {other_name}")
    ax.grid(True, alpha=0.3)

    # Add correlation coefficient
    corr = np.corrcoef(valid_data[reward_col], valid_data[other_col])[0, 1]
    ax.text(
        0.05,
        0.95,
        f"Correlation: {corr:.3f}",
        transform=ax.transAxes,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )


def _plot_reward_statistics(ax: plt.Axes, data: pd.DataFrame, reward_col: str):
    """Plot reward statistics when no comparison data available."""

    clean_values = data[reward_col].dropna()

    if len(clean_values) == 0:
        ax.text(
            0.5,
            0.5,
            "No valid reward data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Reward Statistics")
        return

    # Create box plot
    ax.boxplot(clean_values, vert=False)
    ax.set_xlabel("Reward Value")
    ax.set_title("Reward Statistics")
    ax.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f"""Statistics:
Mean: {clean_values.mean():.3f}
Std: {clean_values.std():.3f}
Min: {clean_values.min():.3f}
Max: {clean_values.max():.3f}
Count: {len(clean_values)}"""

    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        verticalalignment="top",
        fontsize=9,
    )


def generate_drift_plots(
    drift_metrics: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: tuple = (12, 8),
) -> Dict[str, Any]:
    """
    Generate drift analysis plots.

    Args:
        drift_metrics: DataFrame with drift detection results
        output_path: Path to save plots
        figsize: Figure size for plots

    Returns:
        Dictionary with plot information
    """

    if drift_metrics.empty:
        return {"error": "No drift metrics available for plotting"}

    plots_info = {}

    # Set style
    plt.style.use("default")
    sns.set_palette("husl")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Reward Drift Analysis", fontsize=16, fontweight="bold")

    # 1. Drift events over time
    if "step" in drift_metrics.columns and drift_metrics["step"].dtype in [
        "int64",
        "float64",
    ]:
        ax1 = axes[0, 0]
        _plot_drift_timeline(ax1, drift_metrics)
        plots_info["timeline_plot"] = "Drift timeline plot generated"

    # 2. Effect size distribution
    ax2 = axes[0, 1]
    _plot_effect_size_distribution(ax2, drift_metrics)
    plots_info["effect_size_plot"] = "Effect size distribution plot generated"

    # 3. Drift types breakdown
    ax3 = axes[1, 0]
    _plot_drift_types(ax3, drift_metrics)
    plots_info["types_plot"] = "Drift types breakdown plot generated"

    # 4. P-value distribution
    ax4 = axes[1, 1]
    _plot_pvalue_distribution(ax4, drift_metrics)
    plots_info["pvalue_plot"] = "P-value distribution plot generated"

    # Adjust layout
    plt.tight_layout()

    # Save plot if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plots_info["saved_path"] = str(output_path)

    return plots_info


def _plot_drift_timeline(ax: plt.Axes, drift_metrics: pd.DataFrame):
    """Plot drift events over time."""

    step_metrics = drift_metrics[drift_metrics["step"] != "overall"]
    if step_metrics.empty:
        ax.text(
            0.5,
            0.5,
            "No temporal drift data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Drift Timeline")
        return

    # Plot drift events
    ax.scatter(
        step_metrics["step"],
        step_metrics["effect_size"],
        c=step_metrics["p_value"],
        cmap="viridis",
        alpha=0.7,
        s=50,
    )

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Effect Size")
    ax.set_title("Drift Events Timeline")
    ax.grid(True, alpha=0.3)

    # Add colorbar
    scatter = ax.scatter([], [], c=[], cmap="viridis")
    plt.colorbar(scatter, ax=ax, label="P-value")


def _plot_effect_size_distribution(ax: plt.Axes, drift_metrics: pd.DataFrame):
    """Plot distribution of effect sizes."""

    effect_sizes = drift_metrics["effect_size"].dropna()

    if len(effect_sizes) == 0:
        ax.text(
            0.5,
            0.5,
            "No effect size data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Effect Size Distribution")
        return

    ax.hist(effect_sizes, bins=20, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Effect Size (Cohen's d)")
    ax.set_ylabel("Frequency")
    ax.set_title("Effect Size Distribution")
    ax.grid(True, alpha=0.3)

    # Add mean line
    mean_es = effect_sizes.mean()
    ax.axvline(
        mean_es, color="red", linestyle="--", alpha=0.8, label=f"Mean: {mean_es:.3f}"
    )
    ax.legend()


def _plot_drift_types(ax: plt.Axes, drift_metrics: pd.DataFrame):
    """Plot breakdown of drift types."""

    if "drift_type" not in drift_metrics.columns:
        ax.text(
            0.5,
            0.5,
            "No drift type data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Drift Types")
        return

    drift_counts = drift_metrics["drift_type"].value_counts()

    if len(drift_counts) == 0:
        ax.text(
            0.5,
            0.5,
            "No drift type data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Drift Types")
        return

    ax.pie(
        drift_counts.values, labels=drift_counts.index, autopct="%1.1f%%", startangle=90
    )
    ax.set_title("Drift Types Breakdown")


def _plot_pvalue_distribution(ax: plt.Axes, drift_metrics: pd.DataFrame):
    """Plot distribution of p-values."""

    pvalues = drift_metrics["p_value"].dropna()

    if len(pvalues) == 0:
        ax.text(
            0.5,
            0.5,
            "No p-value data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("P-value Distribution")
        return

    ax.hist(pvalues, bins=20, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("P-value")
    ax.set_ylabel("Frequency")
    ax.set_title("P-value Distribution")
    ax.grid(True, alpha=0.3)

    # Add significance threshold
    ax.axvline(0.05, color="red", linestyle="--", alpha=0.8, label="α = 0.05")
    ax.axvline(0.01, color="orange", linestyle="--", alpha=0.8, label="α = 0.01")
    ax.legend()
