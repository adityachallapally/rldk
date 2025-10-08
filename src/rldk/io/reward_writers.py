"""Report writers for reward health analysis."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

try:
    from ..reward.health_analysis import OveroptimizationAnalysis, RewardHealthReport
except ImportError:  # pragma: no cover - avoid circular import during module init
    OveroptimizationAnalysis = None  # type: ignore[assignment]
    RewardHealthReport = Any  # type: ignore[assignment]
from ..reward.length_bias import LengthBiasMetrics
from .unified_writer import UnifiedWriter


def _json_serialize(obj):
    """
    Custom JSON serializer that converts NaN to null.

    Note: This function is deprecated. Use UnifiedWriter._json_serializer instead
    for consistent serialization across the codebase.
    """
    # Handle numpy floating point types and Python float
    if (isinstance(obj, (float, np.floating)) and np.isnan(obj)):
        return None
    # Handle infinity values
    elif (isinstance(obj, (float, np.floating)) and np.isinf(obj)):
        return None
    # Handle numpy integers (convert to Python int for JSON compatibility)
    elif isinstance(obj, np.integer):
        return int(obj)
    # Handle numpy floating point numbers (convert to Python float for JSON compatibility)
    elif isinstance(obj, np.floating):
        return float(obj)
    # Handle pandas NaN values
    elif pd.isna(obj):
        return None
    elif isinstance(obj, dict):
        return {key: _json_serialize(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_json_serialize(item) for item in obj]
    else:
        return obj


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


def write_reward_health_card(report: RewardHealthReport, output_dir: Path) -> None:
    """Write reward health card to markdown file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "reward_health_card.md"

    with open(report_path, "w") as f:
        f.write("# Reward Health Analysis Card\n\n")

        # Overall status
        if report.passed:
            f.write("## ✅ Reward Health Check Passed\n\n")
            f.write(
                "The reward model appears to be healthy with no significant pathologies detected.\n\n"
            )
        else:
            f.write("## 🚨 Reward Health Issues Detected\n\n")
            f.write(
                "The reward model shows potential pathologies that should be investigated.\n\n"
            )

        # Summary of findings
        f.write("## 📊 Summary of Findings\n\n")
        f.write(f"- **Drift Detected:** {'Yes' if report.drift_detected else 'No'}\n")
        f.write(f"- **Saturation Issues:** {len(report.saturation_issues)}\n")
        f.write(f"- **Calibration Score:** {report.calibration_score:.3f}\n")
        f.write(f"- **Shortcut Signals:** {len(report.shortcut_signals)}\n")
        f.write(f"- **Label Leakage Risk:** {report.label_leakage_risk:.3f}\n\n")
        metrics = report.length_bias_metrics
        severity: Optional[float] = None
        if isinstance(metrics, LengthBiasMetrics):
            severity = metrics.bias_severity
        elif isinstance(metrics, dict):
            severity = metrics.get("bias_severity")
        elif metrics is not None and hasattr(metrics, "bias_severity"):
            severity = getattr(metrics, "bias_severity")
        severity_str = f"{severity:.3f}" if severity is not None else "N/A"
        f.write(
            f"- **Length Bias Detected:** {'Yes' if report.length_bias_detected else 'No'}\n"
        )
        f.write(f"- **Length Bias Severity:** {severity_str}\n\n")

        overopt = getattr(report, "overoptimization", None)
        if OveroptimizationAnalysis and isinstance(overopt, OveroptimizationAnalysis):
            if overopt.warning:
                f.write(f"- **Overoptimization Check:** ⚠️ {overopt.warning}\n\n")
            else:
                overopt_status = "Yes" if overopt.flagged else "No"
                f.write(
                    f"- **Overoptimization Flagged:** {overopt_status}\n"
                )
                if overopt.gold_metrics_available:
                    f.write(
                        f"- **Proxy vs Gold Δ:** {overopt.delta:.3f}\n\n"
                    )
                else:
                    f.write("- **Proxy vs Gold Δ:** N/A (no gold metrics)\n\n")
        else:
            f.write("- **Overoptimization Check:** Not evaluated\n\n")

        # Detailed analysis sections
        if report.drift_detected:
            f.write("## 🔄 Reward Drift Analysis\n\n")
            f.write("**Status:** 🚨 Drift detected\n\n")

            if not report.drift_metrics.empty:
                f.write("### Drift Events\n\n")
                f.write("| Step | Type | P-Value | Effect Size |\n")
                f.write("|------|------|---------|-------------|\n")

                for _, row in report.drift_metrics.head(10).iterrows():
                    step = row.get("step", "N/A")
                    drift_type = row.get("drift_type", "N/A")
                    p_value = row.get("p_value", np.nan)
                    effect_size = row.get("effect_size", np.nan)

                    p_str = f"{p_value:.3f}" if not pd.isna(p_value) else "N/A"
                    es_str = f"{effect_size:.3f}" if not pd.isna(effect_size) else "N/A"

                    f.write(f"| {step} | {drift_type} | {p_str} | {es_str} |\n")

                if len(report.drift_metrics) > 10:
                    f.write(
                        f"\n*... and {len(report.drift_metrics) - 10} more drift events*\n"
                    )
            else:
                f.write("No detailed drift metrics available.\n")

        if report.saturation_issues:
            f.write("## 📈 Saturation Analysis\n\n")
            f.write("**Status:** 🚨 Saturation issues detected\n\n")

            for issue in report.saturation_issues:
                f.write(f"- {issue}\n")

            if report.saturation_analysis:
                f.write("\n### Saturation Metrics\n\n")
                for metric, value in report.saturation_analysis.items():
                    if isinstance(value, float):
                        f.write(f"- **{metric}:** {value:.3f}\n")
                    else:
                        f.write(f"- **{metric}:** {value}\n")

        if report.calibration_score < 0.7:
            f.write("## 🎯 Calibration Analysis\n\n")
            f.write("**Status:** ⚠️ Poor calibration detected\n\n")
            f.write(f"**Calibration Score:** {report.calibration_score:.3f}\n\n")

            if report.calibration_details:
                f.write("### Calibration Details\n\n")
                for key, value in report.calibration_details.items():
                    if isinstance(value, dict):
                        f.write(f"#### {key}\n")
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, float):
                                f.write(f"- **{sub_key}:** {sub_value:.3f}\n")
                            else:
                                f.write(f"- **{sub_key}:** {sub_value}\n")
                    elif isinstance(value, float):
                        f.write(f"- **{key}:** {value:.3f}\n")
                    else:
                        f.write(f"- **{key}:** {value}\n")
                    f.write("\n")

        if report.shortcut_signals:
            f.write("## 🚧 Shortcut Signal Detection\n\n")
            f.write("**Status:** 🚨 Shortcut signals detected\n\n")

            for signal in report.shortcut_signals:
                f.write(f"- {signal}\n")

        if report.shortcut_analysis:
            f.write("\n### Shortcut Analysis\n\n")
            for metric, correlation in report.shortcut_analysis.items():
                f.write(f"- **{metric}:** {correlation:.3f}\n")

        if report.label_leakage_risk > 0.3:
            f.write("## 🔒 Label Leakage Analysis\n\n")
            f.write("**Status:** 🚨 Potential label leakage detected\n\n")
            f.write(f"**Risk Score:** {report.label_leakage_risk:.3f}\n\n")
            f.write(
                "The reward model may have access to training signals it shouldn't see.\n"
            )

        if OveroptimizationAnalysis and isinstance(overopt, OveroptimizationAnalysis):
            f.write("## 📉 Reward Overoptimization Watch\n\n")
            if overopt.warning:
                f.write(f"**Status:** ⚠️ {overopt.warning}\n\n")
            else:
                status_line = (
                    "**Status:** 🚨 Potential overoptimization detected\n\n"
                    if overopt.flagged
                    else "**Status:** ✅ No overoptimization warning\n\n"
                )
                f.write(status_line)
                if overopt.gold_metrics_available:
                    f.write(
                        f"- **Proxy Improvement:** {overopt.proxy_improvement:.3f}\n"
                    )
                    f.write(
                        f"- **Gold Improvement:** {overopt.gold_improvement:.3f}\n"
                    )
                    f.write(f"- **Delta:** {overopt.delta:.3f}\n")
                if overopt.correlation_trend:
                    pearson = overopt.correlation_trend.get("pearson_delta")
                    spearman = overopt.correlation_trend.get("spearman_delta")
                    if pearson is not None or spearman is not None:
                        f.write("- **Correlation Trend:**\n")
                        if pearson is not None:
                            f.write(
                                f"  - Pearson Δ: {pearson:.3f}\n"
                            )
                        if spearman is not None:
                            f.write(
                                f"  - Spearman Δ: {spearman:.3f}\n"
                            )
                if overopt.kl_summary:
                    kl_current = overopt.kl_summary.get("kl_current_mean")
                    kl_target = overopt.kl_summary.get("kl_target")
                    if kl_current is not None:
                        f.write(
                            f"- **Recent KL Mean:** {float(kl_current):.3f}\n"
                        )
                    if kl_target is not None:
                        f.write(f"- **KL Target:** {float(kl_target):.3f}\n")
                f.write(
                    f"- **Detector Window:** {overopt.window_size} steps\n"
                )
                f.write(
                    f"- **Delta Threshold:** {overopt.delta_threshold:.3f}\n"
                )
                f.write(
                    f"- **Samples Evaluated:** {overopt.sample_size}\n"
                )
                if overopt.notes:
                    f.write("\n### Notes\n\n")
                    for note in overopt.notes:
                        f.write(f"- {note}\n")

        length_metrics_dict = _length_bias_metrics_to_dict(report.length_bias_metrics)
        if report.length_bias_detected or length_metrics_dict or report.length_bias_recommendations:
            f.write("## 🧵 Length Bias Analysis\n\n")
            status_line = (
                "**Status:** 🚨 Length bias detected\n\n"
                if report.length_bias_detected
                else "**Status:** ✅ No significant length bias detected\n\n"
            )
            f.write(status_line)

            if length_metrics_dict:
                f.write("### Key Metrics\n\n")
                key_fields = {
                    "Valid Samples": length_metrics_dict.get("valid_sample_count"),
                    "Pearson Correlation": length_metrics_dict.get(
                        "pearson_correlation"
                    ),
                    "Spearman Correlation": length_metrics_dict.get(
                        "spearman_correlation"
                    ),
                    "Variance Explained": length_metrics_dict.get(
                        "variance_explained"
                    ),
                    "Bias Severity": length_metrics_dict.get("bias_severity"),
                    "ODIN Reward per Token": length_metrics_dict.get(
                        "odin_reward_per_token"
                    ),
                    "ODIN Optimization Flag": length_metrics_dict.get(
                        "odin_optimization_flag"
                    ),
                }
                for label, value in key_fields.items():
                    formatted = _format_optional_value(value)
                    f.write(f"- **{label}:** {formatted}\n")

                quartiles = length_metrics_dict.get("quartile_metrics") or {}
                if quartiles:
                    f.write("\n### Quartile Summary\n\n")
                    f.write("| Quartile | Length Range | Mean Reward | Count |\n")
                    f.write("|----------|--------------|-------------|-------|\n")
                    for name, values in quartiles.items():
                        length_min = _format_optional_value(values.get("length_min"))
                        length_max = _format_optional_value(values.get("length_max"))
                        reward_mean = _format_optional_value(values.get("mean_reward"))
                        count = _format_optional_value(values.get("count"))
                        f.write(
                            f"| {name.upper()} | {length_min} - {length_max} | {reward_mean} | {count} |\n"
                        )

            if report.length_bias_recommendations:
                f.write("\n### Recommendations\n\n")
                for recommendation in report.length_bias_recommendations:
                    f.write(f"- {recommendation}\n")

        # Recommended fixes
        if report.fixes:
            f.write("## 🔧 Recommended Fixes\n\n")
            for i, fix in enumerate(report.fixes, 1):
                f.write(f"{i}. {fix}\n")

        # Report metadata
        f.write("\n## 📁 Report Location\n\n")
        f.write(f"Full report saved to: `{report_path}`\n")

        if report.drift_metrics is not None and not report.drift_metrics.empty:
            f.write(f"Drift analysis saved to: `{output_dir}/drift_analysis.csv`\n")

        f.write(f"Calibration plots saved to: `{output_dir}/calibration_plots.png`\n")


def write_drift_analysis_csv(report: RewardHealthReport, output_dir: Path) -> None:
    """Write drift analysis results to CSV file."""

    if report.drift_metrics is not None and not report.drift_metrics.empty:
        drift_path = output_dir / "drift_analysis.csv"
        report.drift_metrics.to_csv(drift_path, index=False)


def write_reward_health_summary(report: RewardHealthReport, output_dir: Path) -> None:
    """Write reward health summary as JSON file."""

    summary_path = output_dir / "reward_health_summary.json"

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

    # Use UnifiedWriter for consistent JSON serialization
    writer = UnifiedWriter(output_dir)
    writer.write_json(summary, summary_path.name)


def generate_reward_health_report(
    report: RewardHealthReport,
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

            # Generate calibration plots
            plots_path = output_dir / "calibration_plots.png"
            generate_calibration_plots(
                reward_data=pd.DataFrame(),  # You'd need actual data here
                reward_col="reward_mean",
                output_path=plots_path,
            )

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
