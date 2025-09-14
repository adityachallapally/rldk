"""Log scanning for training runs."""

from pathlib import Path
from typing import Dict, Any, Iterator
import pandas as pd

from rldk.io.readers import read_jsonl, read_tensorboard_export, read_wandb_export
from rldk.adapters.flexible import FlexibleDataAdapter
from rldk.forensics.ppo_scan import scan_ppo_events
from rldk.utils.error_handling import ValidationError, AdapterError


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

    if path.is_file() and path.suffix == ".csv":
        try:
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                yield row.to_dict()
            return
        except Exception:
            pass

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
            
            if data_files[0].suffix == ".jsonl":
                return read_jsonl(data_files[0])

    # Check for log files
    if path.is_dir():
        log_files = list(path.glob("*.log")) + list(path.glob("*.txt"))
        if log_files:
            # Try to parse as JSONL
            return read_jsonl(log_files[0])

    raise ValueError(
        f"Could not detect log format for {path}. "
        f"Supported formats: JSONL, CSV, JSON, Parquet files or directories, "
        f"TensorBoard export directories, W&B export directories. "
        f"For custom formats, ensure data contains standard RL fields like 'step', 'reward', 'kl'."
    )
