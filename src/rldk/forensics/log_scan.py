"""Log scanning for training runs."""

from pathlib import Path
from typing import Dict, Any, Iterator

from rldk.io.readers import read_jsonl, read_tensorboard_export, read_wandb_export
from .ppo_scan import scan_ppo_events


def scan_logs(run_or_export: str) -> Dict[str, Any]:
    """Scan training logs for PPO anomalies and issues."""
    run_or_export = Path(run_or_export)

    if not run_or_export.exists():
        raise FileNotFoundError(f"Path not found: {run_or_export}")

    # Detect log format and read events
    events = detect_and_read_logs(run_or_export)

    # Run PPO scan analysis
    report = scan_ppo_events(events)

    return report


def detect_and_read_logs(path: Path) -> Iterator[Dict[str, Any]]:
    """Detect log format and read events."""

    # Check for TensorBoard export
    if path.is_dir():
        csv_files = list(path.glob("*.csv"))
        if csv_files:
            return read_tensorboard_export(path)

    # Check for Weights & Biases export
    if path.is_dir():
        jsonl_files = list(path.glob("*.jsonl"))
        if jsonl_files:
            return read_wandb_export(path)

    # Check for JSONL file
    if path.is_file() and path.suffix == ".jsonl":
        return read_jsonl(path)

    # Check for JSONL files in directory
    if path.is_dir():
        jsonl_files = list(path.glob("*.jsonl"))
        if jsonl_files:
            # Use the first JSONL file found
            return read_jsonl(jsonl_files[0])

    # Check for log files
    if path.is_dir():
        log_files = list(path.glob("*.log")) + list(path.glob("*.txt"))
        if log_files:
            # Try to parse as JSONL
            return read_jsonl(log_files[0])

    raise ValueError(
        f"Could not detect log format for {path}. "
        f"Expected: TensorBoard export directory, W&B export directory, "
        f"or JSONL file"
    )
