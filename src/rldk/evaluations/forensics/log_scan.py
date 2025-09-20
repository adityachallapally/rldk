"""Log scanning for training runs."""

import json
from pathlib import Path
from typing import Any, Dict, Iterator

import pandas as pd

from rldk.monitoring.adapters.flexible import FlexibleDataAdapter
from rldk.evaluations.forensics.ppo_scan import scan_ppo_events
from rldk.core.io.readers import read_jsonl, read_tensorboard_export, read_wandb_export
from rldk.core.utils.error_handling import AdapterError, ValidationError


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
    """Detect log format and read events with enhanced format support."""

    try:
        adapter = FlexibleDataAdapter(path)
        df = adapter.load()

        for _, row in df.iterrows():
            yield row.to_dict()
        return

    except (ValidationError, AdapterError):
        pass

    # Check for TensorBoard export (CSV files in directory)
    if path.is_dir():
        csv_files = list(path.glob("*.csv"))
        if csv_files:
            yield from read_tensorboard_export(path)
            return

    # Check for Weights & Biases export (JSONL files in directory)
    if path.is_dir():
        jsonl_files = list(path.glob("*.jsonl"))
        if jsonl_files:
            yield from read_wandb_export(path)
            return

    # Check for single JSONL file
    if path.is_file() and path.suffix == ".jsonl":
        yield from read_jsonl(path)
        return

    # Check for single CSV file (fallback if FlexibleDataAdapter failed)
    if path.is_file() and path.suffix == ".csv":
        try:
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                yield row.to_dict()
            return
        except (pd.errors.EmptyDataError, pd.errors.ParserError, FileNotFoundError) as e:
            raise ValueError(f"Failed to parse CSV file {path}: {e}")

    if path.is_dir():
        data_files = (
            list(path.glob("*.jsonl")) +
            list(path.glob("*.csv")) +
            list(path.glob("*.json")) +
            list(path.glob("*.parquet"))
        )

        if data_files:
            try:
                adapter = FlexibleDataAdapter(data_files[0])
                df = adapter.load()
                for _, row in df.iterrows():
                    yield row.to_dict()
                return
            except (ValidationError, AdapterError):
                pass

            first_file = data_files[0]
            if first_file.suffix == ".jsonl":
                yield from read_jsonl(first_file)
                return
            elif first_file.suffix == ".csv":
                try:
                    df = pd.read_csv(first_file)
                    for _, row in df.iterrows():
                        yield row.to_dict()
                    return
                except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                    raise ValueError(f"Failed to parse CSV file {first_file}: {e}")
            elif first_file.suffix == ".json":
                try:
                    with open(first_file) as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        yield from data
                    elif isinstance(data, dict):
                        yield data
                    return
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    raise ValueError(f"Failed to parse JSON file {first_file}: {e}")

    # Check for log files as final fallback
    if path.is_dir():
        log_files = list(path.glob("*.log")) + list(path.glob("*.txt"))
        if log_files:
            # Try to parse as JSONL
            yield from read_jsonl(log_files[0])
            return

    raise ValueError(
        f"Could not detect log format for {path}. "
        f"Supported formats: JSONL, CSV, JSON, Parquet files or directories, "
        f"TensorBoard export directories, W&B export directories. "
        f"For custom formats, ensure data contains standard RL fields like 'step', 'reward', 'kl'."
    )
