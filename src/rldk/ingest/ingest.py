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
        adapter_hint: Optional hint for adapter type ('trl', 'openrlhf', 'wandb', 'custom_jsonl')

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

    # Validate source exists (skip for WandB URIs)
    if not source_str.startswith("wandb://"):
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(
                f"Source path does not exist: {source}\n"
                "Please check the path and ensure the file or directory exists."
            )
    else:
        # For WandB URIs, create a dummy path for compatibility
        source_path = Path(source)

    # Create appropriate adapter with better error messages
    try:
        if adapter_hint == "trl":
            adapter = TRLAdapter(source)
        elif adapter_hint == "openrlhf":
            adapter = OpenRLHFAdapter(source)
        elif adapter_hint == "wandb":
            adapter = WandBAdapter(source)
        elif adapter_hint == "custom_jsonl":
            adapter = CustomJSONLAdapter(source)
        else:
            raise ValueError(
                f"Unknown adapter type: {adapter_hint}\n"
                "Supported adapters: 'trl', 'openrlhf', 'wandb', 'custom_jsonl'"
            )
    except Exception as e:
        raise ValueError(
            f"Failed to create {adapter_hint} adapter for {source}: {e}\n"
            "Please check the source format and try specifying the correct adapter type."
        )

    # Check if adapter can handle the source
    if not adapter.can_handle():
        # Provide detailed error message based on source type
        if source_str.startswith("wandb://"):
            raise ValueError(
                f"Cannot handle {adapter_hint} format for WandB URI: {source}\n"
                f"Expected WandB URI format:\n"
                f"{_get_format_examples('wandb')}\n"
                "Make sure the WandB URI is valid and accessible."
            )
        elif source_path.is_file():
            file_ext = source_path.suffix.lower()
            if file_ext == ".jsonl":
                raise ValueError(
                    f"Cannot handle {adapter_hint} format for file: {source}\n"
                    f"Expected format for {adapter_hint}:\n"
                    f"{_get_format_examples(adapter_hint)}\n"
                    "Try using --adapter custom_jsonl for generic JSONL files."
                )
            else:
                raise ValueError(
                    f"Cannot handle {adapter_hint} format for file: {source}\n"
                    f"File extension '{file_ext}' is not supported by {adapter_hint} adapter.\n"
                    f"Supported extensions: {_get_supported_extensions(adapter_hint)}"
                )
        else:
            raise ValueError(
                f"Cannot handle {adapter_hint} format for directory: {source}\n"
                f"Expected directory structure for {adapter_hint}:\n"
                f"{_get_directory_structure_examples(adapter_hint)}"
            )

    # Load data with robust error handling
    try:
        df = adapter.load()
        logging.info(f"Successfully ingested {len(df)} events from {source}")
    except Exception as e:
        logging.error(f"Failed to ingest {source}: {e}")
        raise ValueError(
            f"Failed to load data from {source} using {adapter_hint} adapter: {e}\n"
            f"Please check the data format and try again.\n"
            f"Expected format: {_get_format_examples(adapter_hint)}"
        )

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

    # Note: WandB URIs are handled in ingest_runs() before this function is called
    # This function only handles local file/directory detection

    if not source_path.exists():
        return "trl"  # Default fallback

    # Check for WandB directory structure (more specific matching)
    # Look for wandb directory name or wandb subdirectory patterns
    if (source_path.name == "wandb" or 
        any(part == "wandb" for part in source_path.parts)):
        wandb_adapter = WandBAdapter(source_path)
        if wandb_adapter.can_handle():
            return "wandb"

    # Check for our custom JSONL format (most specific)
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

    # If it's a JSONL file but no specific format detected, try custom_jsonl
    if source_path.is_file() and source_path.suffix.lower() == ".jsonl":
        return "custom_jsonl"

    # Default to TRL if no specific patterns found
    return "trl"


def _get_format_examples(adapter_type: str) -> str:
    """Get format examples for a specific adapter type."""
    examples = {
        "trl": """TRL format examples:
  JSONL format:
    {"step": 0, "phase": "train", "reward_mean": 0.5, "kl_mean": 0.1, "entropy_mean": 2.0, "loss": 0.3}
    {"step": 1, "phase": "train", "reward_mean": 0.6, "kl_mean": 0.09, "entropy_mean": 1.9, "loss": 0.25}
  
  Log format:
    step: 0, reward: 0.5, kl: 0.1, entropy: 2.0, loss: 0.3
    step: 1, reward: 0.6, kl: 0.09, entropy: 1.9, loss: 0.25""",
        
        "openrlhf": """OpenRLHF format examples:
  JSONL format:
    {"step": 0, "phase": "train", "reward_mean": 0.5, "kl_mean": 0.1, "entropy_mean": 2.0, "loss": 0.3}
    {"step": 1, "phase": "train", "reward_mean": 0.6, "kl_mean": 0.09, "entropy_mean": 1.9, "loss": 0.25}
  
  Log format:
    step: 0, reward: 0.5, kl: 0.1, entropy: 2.0, loss: 0.3
    step: 1, reward: 0.6, kl: 0.09, entropy: 1.9, loss: 0.25""",
        
        "custom_jsonl": """Custom JSONL format examples:
  {"global_step": 0, "reward_scalar": 0.5, "kl_to_ref": 0.1, "entropy": 2.0, "loss": 0.3}
  {"global_step": 1, "reward_scalar": 0.6, "kl_to_ref": 0.09, "entropy": 1.9, "loss": 0.25}
  
  Or with nested structure:
  {"step": 0, "metrics": {"reward": 0.5, "kl": 0.1}, "model_info": {"phase": "train"}}""",
        
        "wandb": """WandB format examples:
  Use wandb:// URI format:
    wandb://project_name/run_id
    wandb://username/project_name/run_id
  
  Or local wandb logs directory:
    ./wandb/run-20240101_120000-abc123/"""
    }
    
    return examples.get(adapter_type, "No format examples available.")


def _get_supported_extensions(adapter_type: str) -> str:
    """Get supported file extensions for a specific adapter type."""
    extensions = {
        "trl": ".jsonl, .log",
        "openrlhf": ".jsonl, .log", 
        "custom_jsonl": ".jsonl",
        "wandb": "wandb:// URI or wandb logs directory"
    }
    
    return extensions.get(adapter_type, "Unknown")


def _get_directory_structure_examples(adapter_type: str) -> str:
    """Get expected directory structure examples for a specific adapter type."""
    structures = {
        "trl": """TRL directory structure:
  training_logs/
    ├── trainer_log.jsonl
    ├── training.log
    └── *_events.jsonl""",
        
        "openrlhf": """OpenRLHF directory structure:
  training_logs/
    ├── training.log
    ├── metrics.jsonl
    └── logs/""",
        
        "custom_jsonl": """Custom JSONL directory structure:
  data/
    ├── metrics.jsonl
    ├── training_data.jsonl
    └── *.jsonl files""",
        
        "wandb": """WandB directory structure:
  wandb/
    └── run-20240101_120000-abc123/
        ├── files/
        ├── logs/
        └── config.yaml"""
    }
    
    return structures.get(adapter_type, "No structure examples available.")
