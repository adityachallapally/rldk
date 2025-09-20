"""Writers for reports and visualizations."""

from pathlib import Path
from typing import Any, Dict, Union

import matplotlib.pyplot as plt


def mkdir_reports() -> Path:
    """Create rldk_reports directory and return path."""
    reports_dir = Path("rldk_reports")
    reports_dir.mkdir(exist_ok=True)
    return reports_dir


def write_json(report: Dict[str, Any], path: Union[str, Path]) -> None:
    """Write report dictionary to JSON file."""
    path = Path(path)

    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Use UnifiedWriter for consistent JSON serialization
    from .unified_writer import UnifiedWriter
    writer = UnifiedWriter(path.parent)
    writer.write_json(report, path.name)


def write_png(fig: plt.Figure, path: Union[str, Path]) -> None:
    """Save matplotlib figure as PNG."""
    path = Path(path)

    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(path, dpi=150, bbox_inches="tight")


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
    write_json(card_data, output_path / "drift_card.json")

    # Write markdown format
    md_content = _generate_drift_card_md(card_data)
    with open(output_path / "drift_card.md", "w") as f:
        f.write(md_content)


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
