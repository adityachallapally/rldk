"""Main ingest function for training runs."""

from pathlib import Path
from typing import Union, Optional, List
import pandas as pd

from ..adapters import TRLAdapter, OpenRLHFAdapter, WandBAdapter, CustomJSONLAdapter
from ..io.event_schema import Event, dataframe_to_events


def ingest_runs(
    source: Union[str, Path], adapter_hint: Optional[str] = None
) -> pd.DataFrame:
    """
    Ingest training runs from various sources.

    Args:
        source: Path to logs directory, file, or wandb:// URI
        adapter_hint: Optional hint for adapter type ('trl', 'openrlhf', 'wandb')

    Returns:
        DataFrame with standardized training metrics
    """
    import logging
    
    source_str = str(source)

    # Try to auto-detect adapter if no hint provided
    if adapter_hint is None:
        if source_str.startswith("wandb://"):
            adapter_hint = "wandb"
        else:
            # Try to detect from file content
            adapter_hint = _detect_adapter_type(source)

    # Create appropriate adapter
    if adapter_hint == "trl":
        adapter = TRLAdapter(source)
    elif adapter_hint == "openrlhf":
        adapter = OpenRLHFAdapter(source)
    elif adapter_hint == "wandb":
        adapter = WandBAdapter(source)
    elif adapter_hint == "custom_jsonl":
        adapter = CustomJSONLAdapter(source)
    else:
        raise ValueError(f"Unknown adapter type: {adapter_hint}")

    # Load data with robust error handling
    try:
        df = adapter.load()
        logging.info(f"Successfully ingested {len(df)} events from {source}")
    except Exception as e:
        logging.error(f"Failed to ingest {source}: {e}")
        raise

    # Validate schema
    required_cols = [
        "step",
        "phase",
        "reward_mean",
        "reward_std",
        "kl_mean",
        "entropy_mean",
        "clip_frac",
        "grad_norm",
        "lr",
        "loss",
        "tokens_in",
        "tokens_out",
        "wall_time",
        "seed",
        "run_id",
        "git_sha",
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    # Ensure step column is present and numeric
    if "step" not in df.columns or df["step"].isna().all():
        df["step"] = range(len(df))

    # Sort by step
    df = df.sort_values("step").reset_index(drop=True)

    return df[required_cols]


def ingest_runs_to_events(
    source: Union[str, Path], adapter_hint: Optional[str] = None
) -> List[Event]:
    """
    Ingest training runs and convert to normalized Event objects.

    Args:
        source: Path to logs directory, file, or wandb:// URI
        adapter_hint: Optional hint for adapter type ('trl', 'openrlhf', 'wandb')

    Returns:
        List of Event objects with normalized training data
    """
    # Get the DataFrame first
    df = ingest_runs(source, adapter_hint)

    # Extract run_id from the data
    run_id = (
        df["run_id"].iloc[0]
        if "run_id" in df.columns and not df["run_id"].isna().all()
        else str(source)
    )
    git_sha = (
        df["git_sha"].iloc[0]
        if "git_sha" in df.columns and not df["git_sha"].isna().all()
        else None
    )

    # Convert to events
    events = dataframe_to_events(df, run_id, git_sha)

    return events


def _detect_adapter_type(source: Union[str, Path]) -> str:
    """Auto-detect adapter type from source content."""
    source_path = Path(source)

    if not source_path.exists():
        return "trl"  # Default fallback

    # Check for our custom JSONL format first
    custom_adapter = CustomJSONLAdapter(source_path)
    if custom_adapter.can_handle():
        return "custom_jsonl"

    # Check for TRL-specific patterns
    trl_adapter = TRLAdapter(source_path)
    if trl_adapter.can_handle():
        return "trl"

    # Check for OpenRLHF-specific patterns
    openrlhf_adapter = OpenRLHFAdapter(source_path)
    if openrlhf_adapter.can_handle():
        return "openrlhf"

    # Default to TRL if no specific patterns found
    return "trl"
