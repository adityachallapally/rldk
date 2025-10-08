"""Consolidated writers for all report types and data formats."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .consolidated_schemas import Event, MetricsSchema
from .unified_writer import UnifiedWriter
from ..reward.length_bias import LengthBiasMetrics

try:
    from ..reward.health_analysis import OveroptimizationAnalysis, RewardHealthReport
except ImportError:  # pragma: no cover - avoid circular import during module init
    OveroptimizationAnalysis = None  # type: ignore[assignment]
    RewardHealthReport = Any  # type: ignore[assignment]


def _length_bias_metrics_to_dict(
    metrics: Optional[LengthBiasMetrics],
) -> Dict[str, Any]:
    if metrics is None:
        return {}
    if isinstance(metrics, LengthBiasMetrics):
        return metrics.to_dict()
    if isinstance(metrics, dict):
        return metrics
    if hasattr(metrics, "to_dict"):
        return metrics.to_dict()
    return {}


def _format_optional_value(value: Any) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "N/A"
        return f"{float(value):.3f}"
    if value is None:
        return "N/A"
    return str(value)


def write_drift_card(drift_data: Dict[str, Any], output_dir: Union[str, Path]) -> None:
    """Write drift card to both JSON and markdown formats.

    Args:
        drift_data: Dictionary containing drift analysis data with keys like:
            - diverged: bool indicating if divergence was detected
            - first_step: int step where divergence first occurred
            - tripped_signals: list of signals that triggered
            - signals_monitored: list of all signals monitored
            - k_consecutive: number of consecutive violations required
            - window_size: rolling window size for analysis
            - tolerance: z-score threshold used
        output_dir: Directory to write the drift card files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Ensure required keys are present for compatibility with tests
    card_data = drift_data.copy()
    if "tripped_signals" not in card_data:
        card_data["tripped_signals"] = []

    # Write JSON format
    writer = UnifiedWriter(output_path)
    writer.write_json(card_data, "drift_card.json")

    # Write markdown format
    md_content = _generate_drift_card_md(card_data)
    writer.write_markdown(md_content, "drift_card.md")


def _generate_drift_card_md(drift_data: Dict[str, Any]) -> str:
    """Generate markdown content for drift card."""
    lines = []
    lines.append("# Drift Detection Card")
    lines.append("")

    if drift_data.get("diverged", False):
        lines.append("## 🚨 Drift Detected")
        lines.append("")
        first_step = drift_data.get("first_step", "unknown")
        lines.append(f"Divergence detected at step {first_step}.")

        tripped_signals = drift_data.get("tripped_signals", [])
        if tripped_signals:
            lines.append("")
            lines.append("### Tripped Signals")
            for signal in tripped_signals:
                lines.append(f"- {signal}")
    else:
        lines.append("## ✅ No Drift Detected")
        lines.append("")
        lines.append("The runs appear to be consistent within the specified tolerance.")

    lines.append("")
    lines.append("## 📁 Report Location")
    lines.append("")
    lines.append(
        f"Full report saved to: `{drift_data.get('output_path', 'drift_card.md')}`"
    )

    lines.append("")
    lines.append("## 🔍 Analysis Parameters")
    lines.append("")

    if "signals_monitored" in drift_data:
        signals = drift_data["signals_monitored"]
        if isinstance(signals, list):
            signals_str = ", ".join(signals)
        else:
            signals_str = str(signals)
        lines.append(f"- **Signals monitored:** {signals_str}")

    if "tolerance" in drift_data:
        lines.append(f"- **Tolerance:** {drift_data['tolerance']}")

    if "k_consecutive" in drift_data:
        lines.append(
            f"- **Consecutive violations required:** {drift_data['k_consecutive']}"
        )

    if "window_size" in drift_data:
        lines.append(f"- **Window size:** {drift_data['window_size']}")

    if drift_data.get("diverged", False):
        total_events = len(drift_data.get("tripped_signals", []))
    else:
        total_events = 0
    lines.append(f"- **Total divergence events:** {total_events}")

    return "\n".join(lines)


def write_reward_health_card(report, output_dir: Path) -> None:
    """Write reward health card to markdown file."""
    writer = UnifiedWriter(output_dir)

    content = _generate_reward_health_card_md(report)
    writer.write_markdown(content, "reward_health_card.md")


def _generate_reward_health_card_md(report) -> str:
    """Generate markdown content for reward health card."""
    lines = []
    lines.append("# Reward Health Analysis Card\n")

    # Overall status
    if report.passed:
        lines.append("## ✅ Reward Health Check Passed\n")
        lines.append(
            "The reward model appears to be healthy with no significant pathologies detected.\n"
        )
    else:
        lines.append("## 🚨 Reward Health Issues Detected\n")
        lines.append(
            "The reward model shows potential pathologies that should be investigated.\n"
        )

    # Summary of findings
    lines.append("## 📊 Summary of Findings\n")
    lines.append(f"- **Drift Detected:** {'Yes' if report.drift_detected else 'No'}\n")
    lines.append(f"- **Saturation Issues:** {len(report.saturation_issues)}\n")
    lines.append(f"- **Calibration Score:** {report.calibration_score:.3f}\n")
    lines.append(f"- **Shortcut Signals:** {len(report.shortcut_signals)}\n")
    lines.append(f"- **Label Leakage Risk:** {report.label_leakage_risk:.3f}\n")
    metrics = report.length_bias_metrics
    severity: Optional[float] = None
    if isinstance(metrics, LengthBiasMetrics):
        severity = metrics.bias_severity
    elif isinstance(metrics, dict):
        severity = metrics.get("bias_severity")
    elif metrics is not None and hasattr(metrics, "bias_severity"):
        severity = getattr(metrics, "bias_severity")
    severity_str = f"{severity:.3f}" if severity is not None else "N/A"
    lines.append(
        f"- **Length Bias Detected:** {'Yes' if report.length_bias_detected else 'No'}\n"
    )
    lines.append(f"- **Length Bias Severity:** {severity_str}\n")

    overopt = getattr(report, "overoptimization", None)
    if OveroptimizationAnalysis and isinstance(overopt, OveroptimizationAnalysis):
        if overopt.warning:
            lines.append(f"- **Overoptimization Check:** ⚠️ {overopt.warning}\n\n")
        else:
            lines.append(
                f"- **Overoptimization Flagged:** {'Yes' if overopt.flagged else 'No'}\n"
            )
            if overopt.gold_metrics_available:
                lines.append(f"- **Proxy vs Gold Δ:** {overopt.delta:.3f}\n\n")
            else:
                lines.append("- **Proxy vs Gold Δ:** N/A (no gold metrics)\n\n")
    else:
        lines.append("- **Overoptimization Check:** Not evaluated\n\n")

    # Detailed analysis sections
    if report.drift_detected:
        lines.append("## 🔄 Reward Drift Analysis\n")
        lines.append("**Status:** 🚨 Drift detected\n")

        if not report.drift_metrics.empty:
            lines.append("### Drift Events\n")
            lines.append("| Step | Type | P-Value | Effect Size |\n")
            lines.append("|------|------|---------|-------------|\n")

            for _, row in report.drift_metrics.head(10).iterrows():
                step = row.get("step", "N/A")
                drift_type = row.get("drift_type", "N/A")
                p_value = row.get("p_value", np.nan)
                effect_size = row.get("effect_size", np.nan)

                p_str = f"{p_value:.3f}" if not pd.isna(p_value) else "N/A"
                es_str = f"{effect_size:.3f}" if not pd.isna(effect_size) else "N/A"

                lines.append(f"| {step} | {drift_type} | {p_str} | {es_str} |\n")

            if len(report.drift_metrics) > 10:
                lines.append(
                    f"\n*... and {len(report.drift_metrics) - 10} more drift events*\n"
                )
        else:
            lines.append("No detailed drift metrics available.\n")

    if report.saturation_issues:
        lines.append("## 📈 Saturation Analysis\n")
        lines.append("**Status:** 🚨 Saturation issues detected\n")

        for issue in report.saturation_issues:
            lines.append(f"- {issue}\n")

        if report.saturation_analysis:
            lines.append("\n### Saturation Metrics\n")
            for metric, value in report.saturation_analysis.items():
                if isinstance(value, float):
                    lines.append(f"- **{metric}:** {value:.3f}\n")
                else:
                    lines.append(f"- **{metric}:** {value}\n")

    if report.calibration_score < 0.7:
        lines.append("## 🎯 Calibration Analysis\n")
        lines.append("**Status:** ⚠️ Poor calibration detected\n")
        lines.append(f"**Calibration Score:** {report.calibration_score:.3f}\n")

        if report.calibration_details:
            lines.append("### Calibration Details\n")
            for key, value in report.calibration_details.items():
                if isinstance(value, dict):
                    lines.append(f"#### {key}\n")
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, float):
                            lines.append(f"- **{sub_key}:** {sub_value:.3f}\n")
                        else:
                            lines.append(f"- **{sub_key}:** {sub_value}\n")
                elif isinstance(value, float):
                    lines.append(f"- **{key}:** {value:.3f}\n")
                else:
                    lines.append(f"- **{key}:** {value}\n")
                lines.append("\n")

    if report.shortcut_signals:
        lines.append("## 🚧 Shortcut Signal Detection\n")
        lines.append("**Status:** 🚨 Shortcut signals detected\n")

        for signal in report.shortcut_signals:
            lines.append(f"- {signal}\n")

        if report.shortcut_analysis:
            lines.append("\n### Shortcut Analysis\n")
            for metric, correlation in report.shortcut_analysis.items():
                lines.append(f"- **{metric}:** {correlation:.3f}\n")

    if report.label_leakage_risk > 0.3:
        lines.append("## 🔒 Label Leakage Analysis\n")
        lines.append("**Status:** 🚨 Potential label leakage detected\n")
        lines.append(f"**Risk Score:** {report.label_leakage_risk:.3f}\n")
        lines.append(
            "The reward model may have access to training signals it shouldn't see.\n"
        )

    if OveroptimizationAnalysis and isinstance(overopt, OveroptimizationAnalysis):
        lines.append("## 📉 Reward Overoptimization Watch\n")
        if overopt.warning:
            lines.append(f"**Status:** ⚠️ {overopt.warning}\n\n")
        else:
            status_line = (
                "**Status:** 🚨 Potential overoptimization detected\n\n"
                if overopt.flagged
                else "**Status:** ✅ No overoptimization warning\n\n"
            )
            lines.append(status_line)
            if overopt.gold_metrics_available:
                lines.append(
                    f"- **Proxy Improvement:** {overopt.proxy_improvement:.3f}\n"
                )
                lines.append(
                    f"- **Gold Improvement:** {overopt.gold_improvement:.3f}\n"
                )
                lines.append(f"- **Delta:** {overopt.delta:.3f}\n")
            if overopt.correlation_trend:
                pearson = overopt.correlation_trend.get("pearson_delta")
                spearman = overopt.correlation_trend.get("spearman_delta")
                if pearson is not None or spearman is not None:
                    lines.append("- **Correlation Trend:**\n")
                    if pearson is not None:
                        lines.append(f"  - Pearson Δ: {pearson:.3f}\n")
                    if spearman is not None:
                        lines.append(f"  - Spearman Δ: {spearman:.3f}\n")
            if overopt.kl_summary:
                kl_current = overopt.kl_summary.get("kl_current_mean")
                kl_target = overopt.kl_summary.get("kl_target")
                if kl_current is not None:
                    lines.append(f"- **Recent KL Mean:** {float(kl_current):.3f}\n")
                if kl_target is not None:
                    lines.append(f"- **KL Target:** {float(kl_target):.3f}\n")
            lines.append(f"- **Detector Window:** {overopt.window_size} steps\n")
            lines.append(f"- **Delta Threshold:** {overopt.delta_threshold:.3f}\n")
            lines.append(f"- **Samples Evaluated:** {overopt.sample_size}\n")
            if overopt.notes:
                lines.append("\n### Notes\n")
                for note in overopt.notes:
                    lines.append(f"- {note}\n")

    length_metrics_dict = _length_bias_metrics_to_dict(report.length_bias_metrics)
    if report.length_bias_detected or length_metrics_dict or report.length_bias_recommendations:
        lines.append("## 🧵 Length Bias Analysis\n")
        status_line = (
            "**Status:** 🚨 Length bias detected\n"
            if report.length_bias_detected
            else "**Status:** ✅ No significant length bias detected\n"
        )
        lines.append(status_line)

        if length_metrics_dict:
            lines.append("### Key Metrics\n")
            key_fields = {
                "Valid Samples": length_metrics_dict.get("valid_sample_count"),
                "Pearson Correlation": length_metrics_dict.get("pearson_correlation"),
                "Spearman Correlation": length_metrics_dict.get("spearman_correlation"),
                "Variance Explained": length_metrics_dict.get("variance_explained"),
                "Bias Severity": length_metrics_dict.get("bias_severity"),
                "ODIN Reward per Token": length_metrics_dict.get(
                    "odin_reward_per_token"
                ),
                "ODIN Optimization Flag": length_metrics_dict.get(
                    "odin_optimization_flag"
                ),
            }
            for label, value in key_fields.items():
                lines.append(f"- **{label}:** {_format_optional_value(value)}\n")

            quartiles = length_metrics_dict.get("quartile_metrics") or {}
            if quartiles:
                lines.append("\n### Quartile Summary\n")
                lines.append("| Quartile | Length Range | Mean Reward | Count |\n")
                lines.append("|----------|--------------|-------------|-------|\n")
                for name, values in quartiles.items():
                    length_min = _format_optional_value(values.get("length_min"))
                    length_max = _format_optional_value(values.get("length_max"))
                    reward_mean = _format_optional_value(values.get("mean_reward"))
                    count = _format_optional_value(values.get("count"))
                    lines.append(
                        f"| {name.upper()} | {length_min} - {length_max} | {reward_mean} | {count} |\n"
                    )

        if report.length_bias_recommendations:
            lines.append("\n### Recommendations\n")
            for recommendation in report.length_bias_recommendations:
                lines.append(f"- {recommendation}\n")

    # Recommended fixes
    if report.fixes:
        lines.append("## 🔧 Recommended Fixes\n")
        for i, fix in enumerate(report.fixes, 1):
            lines.append(f"{i}. {fix}\n")

    # Report metadata
    lines.append("\n## 📁 Report Location\n")
    lines.append("Full report saved to: `reward_health_card.md`\n")

    if report.drift_metrics is not None and not report.drift_metrics.empty:
        lines.append("Drift analysis saved to: `drift_analysis.csv`\n")

    lines.append("Calibration plots saved to: `calibration_plots.png`\n")

    return "".join(lines)


def write_drift_analysis_csv(report, output_dir: Path) -> None:
    """Write drift analysis results to CSV file."""
    if report.drift_metrics is not None and not report.drift_metrics.empty:
        writer = UnifiedWriter(output_dir)
        writer.write_csv(report.drift_metrics, "drift_analysis.csv")


def write_reward_health_summary(report, output_dir: Path) -> None:
    """Write reward health summary as JSON file."""
    writer = UnifiedWriter(output_dir)

    summary = {
        "passed": report.passed,
        "overall_status": "passed" if report.passed else "failed",
        "drift_detected": report.drift_detected,
        "saturation_issues_count": len(report.saturation_issues),
        "calibration_score": report.calibration_score,
        "shortcut_signals_count": len(report.shortcut_signals),
        "label_leakage_risk": report.label_leakage_risk,
        "fixes": report.fixes,
        "saturation_analysis": report.saturation_analysis,
        "shortcut_analysis": report.shortcut_analysis,
        "length_bias_detected": report.length_bias_detected,
        "length_bias_recommendations": report.length_bias_recommendations,
        "length_bias_metrics": _length_bias_metrics_to_dict(
            report.length_bias_metrics
        ),
    }

    overopt = getattr(report, "overoptimization", None)
    if OveroptimizationAnalysis and isinstance(overopt, OveroptimizationAnalysis):
        summary["overoptimization"] = overopt.to_dict()
    elif hasattr(overopt, "to_dict"):
        summary["overoptimization"] = overopt.to_dict()
    else:
        summary["overoptimization"] = {}

    # Add drift summary if available
    if report.drift_metrics is not None and not report.drift_metrics.empty:
        summary["drift_summary"] = {
            "total_events": len(report.drift_metrics),
            "drift_types": (
                report.drift_metrics["drift_type"].value_counts().to_dict()
                if "drift_type" in report.drift_metrics.columns
                else {}
            ),
            "mean_effect_size": (
                report.drift_metrics["effect_size"].mean()
                if "effect_size" in report.drift_metrics.columns
                else np.nan
            ),
            "max_effect_size": (
                report.drift_metrics["effect_size"].max()
                if "effect_size" in report.drift_metrics.columns
                else np.nan
            ),
        }

    # Add calibration details if available
    if report.calibration_details:
        summary["calibration_details"] = report.calibration_details

    writer.write_json(summary, "reward_health_summary.json")


def generate_reward_health_report(
    report,
    output_dir: Union[str, Path],
    generate_plots: bool = True,
) -> None:
    """
    Generate comprehensive reward health report.

    Args:
        report: RewardHealthReport object
        output_dir: Directory to save reports
        generate_plots: Whether to generate calibration plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write main report
    write_reward_health_card(report, output_dir)

    # Write CSV data
    write_drift_analysis_csv(report, output_dir)

    # Write JSON summary
    write_reward_health_summary(report, output_dir)

    # Generate plots if requested
    if generate_plots:
        try:
            from ..reward.plots import generate_calibration_plots

            # Generate calibration plots using actual reward data from the report
            writer = UnifiedWriter(output_dir)

            # Extract reward data from the report if available
            reward_data = None
            if hasattr(report, 'reward_data') and report.reward_data is not None:
                reward_data = report.reward_data
            elif hasattr(report, 'metrics') and report.metrics is not None:
                # Try to extract reward data from metrics
                if hasattr(report.metrics, 'to_dataframe'):
                    reward_data = report.metrics.to_dataframe()
                elif isinstance(report.metrics, pd.DataFrame):
                    reward_data = report.metrics
            elif hasattr(report, 'drift_metrics') and report.drift_metrics is not None:
                # Use drift metrics as fallback if available
                reward_data = report.drift_metrics

            # Only generate plots if we have actual data
            if reward_data is not None and not reward_data.empty:
                # Determine the appropriate reward column
                reward_col = "reward_mean"
                if "reward_mean" not in reward_data.columns:
                    # Look for alternative reward columns
                    reward_cols = [col for col in reward_data.columns if 'reward' in col.lower()]
                    if reward_cols:
                        reward_col = reward_cols[0]
                    else:
                        print("Warning: No reward columns found in data, skipping plot generation")
                        return

                generate_calibration_plots(
                    reward_data=reward_data,
                    reward_col=reward_col,
                    output_path=writer.output_dir / "calibration_plots.png",
                )
            else:
                print("Warning: No reward data available for plot generation")

        except Exception as e:
            # Log plot generation failure but continue
            print(f"Warning: Failed to generate plots: {e}")

    print(f"Reward health report generated in: {output_dir}")
    print(f"- Health card: {output_dir}/reward_health_card.md")
    print(f"- Summary: {output_dir}/reward_health_summary.json")
    if report.drift_metrics is not None and not report.drift_metrics.empty:
        print(f"- Drift analysis: {output_dir}/drift_analysis.csv")
    if generate_plots:
        print(f"- Calibration plots: {output_dir}/calibration_plots.png")


def write_events_jsonl(
    events: List[Event],
    file_path: Union[str, Path]
) -> None:
    """
    Write events to JSONL file with consistent data validation.

    Args:
        events: List of Event objects
        file_path: Output file path
    """
    file_path = Path(file_path)
    writer = UnifiedWriter(file_path.parent)

    # Convert events to dictionaries with consistent serialization
    event_data = []
    for event in events:
        event_dict = event.to_dict()
        # Ensure consistent handling of nested data structures
        event_data.append(event_dict)

    writer.write_jsonl(event_data, file_path.name)


def write_metrics_jsonl(df: pd.DataFrame, file_path: Union[str, Path]) -> None:
    """
    Write metrics DataFrame to JSONL file with consistent schema validation.

    Args:
        df: DataFrame with training metrics
        file_path: Output file path
    """
    file_path = Path(file_path)
    writer = UnifiedWriter(file_path.parent)

    # Always use MetricsSchema for consistent validation and formatting
    writer.write_metrics_jsonl(df, file_path.name, MetricsSchema)


def write_determinism_card(
    card_data: Dict[str, Any],
    output_dir: Union[str, Path]
) -> None:
    """Write determinism card to JSON file."""
    output_dir = Path(output_dir)
    writer = UnifiedWriter(output_dir)
    writer.write_json(card_data, "determinism_card.json")


def write_ppo_scan_report(
    report_data: Dict[str, Any],
    output_dir: Union[str, Path]
) -> None:
    """Write PPO scan report to JSON file."""
    output_dir = Path(output_dir)
    writer = UnifiedWriter(output_dir)
    writer.write_json(report_data, "ppo_scan.json")


def write_ckpt_diff_report(
    report_data: Dict[str, Any],
    output_dir: Union[str, Path]
) -> None:
    """Write checkpoint diff report to JSON file."""
    output_dir = Path(output_dir)
    writer = UnifiedWriter(output_dir)
    writer.write_json(report_data, "ckpt_diff.json")


def write_reward_drift_report(
    report_data: Dict[str, Any],
    output_dir: Union[str, Path]
) -> None:
    """Write reward drift report to JSON file."""
    output_dir = Path(output_dir)
    writer = UnifiedWriter(output_dir)
    writer.write_json(report_data, "reward_drift.json")


def write_run_comparison(
    comparison_data: Dict[str, Any],
    output_dir: Union[str, Path]
) -> None:
    """Write run comparison report to JSON file."""
    output_dir = Path(output_dir)
    writer = UnifiedWriter(output_dir)
    writer.write_json(comparison_data, "run_comparison.json")


def write_eval_summary(
    eval_data: Dict[str, Any],
    output_dir: Union[str, Path]
) -> None:
    """Write evaluation summary to JSON file."""
    output_dir = Path(output_dir)
    writer = UnifiedWriter(output_dir)
    writer.write_json(eval_data, "eval_summary.json")


def write_environment_audit(
    audit_data: Dict[str, Any],
    output_dir: Union[str, Path]
) -> None:
    """Write environment audit report to JSON file."""
    output_dir = Path(output_dir)
    writer = UnifiedWriter(output_dir)
    writer.write_json(audit_data, "env_audit.json")


def write_replay_comparison(
    comparison_data: Dict[str, Any],
    output_dir: Union[str, Path]
) -> None:
    """Write replay comparison report to JSON file."""
    output_dir = Path(output_dir)
    writer = UnifiedWriter(output_dir)
    writer.write_json(comparison_data, "replay_comparison.json")


def write_tracking_data(
    tracking_data: Dict[str, Any],
    output_dir: Union[str, Path]
) -> None:
    """Write tracking data to JSON file."""
    output_dir = Path(output_dir)
    writer = UnifiedWriter(output_dir)
    writer.write_json(tracking_data, "tracking_data.json")


# Backward compatibility functions
def mkdir_reports() -> Path:
    """Create rldk_reports directory and return path (backward compatibility)."""
    reports_dir = Path("rldk_reports")
    reports_dir.mkdir(exist_ok=True)
    return reports_dir


def write_json(report: Dict[str, Any], path: Union[str, Path]) -> None:
    """Write report dictionary to JSON file (backward compatibility)."""
    path = Path(path)
    writer = UnifiedWriter(path.parent)
    writer.write_json(report, path.name)


def write_png(fig: plt.Figure, path: Union[str, Path]) -> None:
    """Save matplotlib figure as PNG (backward compatibility)."""
    path = Path(path)
    writer = UnifiedWriter(path.parent)
    writer.write_png(fig, path.name)
