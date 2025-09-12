"""Main ingest function for training runs."""

from pathlib import Path
from typing import Union, Optional, List
import pandas as pd
import logging

from ..adapters import TRLAdapter, OpenRLHFAdapter, WandBAdapter, CustomJSONLAdapter
from ..io.event_schema import Event, dataframe_to_events
from ..utils.error_handling import (
    AdapterError, ValidationError, format_error_message, 
    validate_file_path, validate_data_format, safe_operation
)
from ..utils.progress import progress_bar, spinner, timed_operation_context


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
        
    Raises:
        ValidationError: If source validation fails
        AdapterError: If data loading fails
    """
    source_str = str(source)
    logger = logging.getLogger(__name__)

    # Validate source format
    try:
        if source_str.startswith("wandb://"):
            # Validate wandb URI format
            parts = source_str[8:].split("/")
            if len(parts) != 3:
                raise ValidationError(
                    f"Invalid wandb URI format: {source_str}",
                    suggestion="Use format: wandb://entity/project/run_id",
                    error_code="INVALID_WANDB_URI"
                )
        else:
            # Validate file/directory path
            source_path = validate_file_path(source, must_exist=True)
            if source_path.is_file():
                validate_file_path(source, file_extensions=[".jsonl", ".log"])
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(
            f"Failed to validate source: {e}",
            suggestion="Check that the source path exists and is accessible",
            error_code="SOURCE_VALIDATION_FAILED"
        ) from e

    # Try to auto-detect adapter if no hint provided
    if adapter_hint is None:
        try:
            adapter_hint = _detect_adapter_type(source)
            logger.info(f"Auto-detected adapter: {adapter_hint}")
        except Exception as e:
            raise AdapterError(
                f"Failed to auto-detect adapter type: {e}",
                suggestion="Specify adapter type explicitly with --adapter",
                error_code="ADAPTER_DETECTION_FAILED"
            ) from e

    # Validate adapter type
    valid_adapters = ["trl", "openrlhf", "wandb", "custom_jsonl"]
    if adapter_hint not in valid_adapters:
        raise ValidationError(
            f"Invalid adapter type: {adapter_hint}",
            suggestion=f"Use one of: {', '.join(valid_adapters)}",
            error_code="INVALID_ADAPTER_TYPE"
        )

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
            raise ValidationError(
                f"Unknown adapter type: {adapter_hint}",
                suggestion=f"Use one of: {', '.join(valid_adapters)}",
                error_code="UNKNOWN_ADAPTER_TYPE"
            )
    except Exception as e:
        raise AdapterError(
            f"Failed to create adapter: {e}",
            suggestion="Check that the adapter type is correct and dependencies are installed",
            error_code="ADAPTER_CREATION_FAILED"
        ) from e

    # Check if adapter can handle the source
    if not adapter.can_handle():
        # Provide detailed error message based on source type
        if source_str.startswith("wandb://"):
            raise AdapterError(
                f"Adapter '{adapter_hint}' cannot handle WandB URI: {source}",
                suggestion=f"Expected WandB URI format:\n{_get_format_examples('wandb')}\nMake sure the WandB URI is valid and accessible.",
                error_code="ADAPTER_CANNOT_HANDLE_SOURCE",
                details={"adapter": adapter_hint, "source": str(source)}
            )
        elif source_path.is_file():
            file_ext = source_path.suffix.lower()
            if file_ext == ".jsonl":
                raise AdapterError(
                    f"Adapter '{adapter_hint}' cannot handle file: {source}",
                    suggestion=f"Expected format for {adapter_hint}:\n{_get_format_examples(adapter_hint)}\nTry using --adapter custom_jsonl for generic JSONL files.",
                    error_code="ADAPTER_CANNOT_HANDLE_SOURCE",
                    details={"adapter": adapter_hint, "source": str(source)}
                )
            else:
                raise AdapterError(
                    f"Adapter '{adapter_hint}' cannot handle file: {source}",
                    suggestion=f"File extension '{file_ext}' is not supported by {adapter_hint} adapter.\nSupported extensions: {_get_supported_extensions(adapter_hint)}",
                    error_code="ADAPTER_CANNOT_HANDLE_SOURCE",
                    details={"adapter": adapter_hint, "source": str(source)}
                )
        else:
            raise AdapterError(
                f"Adapter '{adapter_hint}' cannot handle directory: {source}",
                suggestion=f"Expected directory structure for {adapter_hint}:\n{_get_directory_structure_examples(adapter_hint)}",
                error_code="ADAPTER_CANNOT_HANDLE_SOURCE",
                details={"adapter": adapter_hint, "source": str(source)}
            )

    # Load data with robust error handling
    try:
        with spinner(f"Loading data with {adapter_hint} adapter"):
            df = adapter.load()
            
        if df.empty:
            raise AdapterError(
                "No data found in source",
                suggestion="Check that the source contains valid training data",
                error_code="NO_DATA_FOUND"
            )
            
        logger.info(f"Successfully ingested {len(df)} events from {source}")
        
    except AdapterError:
        raise
    except Exception as e:
        raise AdapterError(
            f"Failed to load data: {e}",
            suggestion="Check that the source contains valid data in the expected format",
            error_code="DATA_LOAD_FAILED",
            details={"adapter": adapter_hint, "source": str(source)}
        ) from e

    # Validate and standardize schema
    try:
        df = _standardize_schema(df)
        logger.info(f"Schema standardized, {len(df)} records ready")
    except Exception as e:
        raise AdapterError(
            f"Failed to standardize schema: {e}",
            suggestion="Check that the data contains the required fields",
            error_code="SCHEMA_STANDARDIZATION_FAILED"
        ) from e

    return df


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


def _standardize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize DataFrame schema to required format."""
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

    # Add missing columns with None values
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    # Ensure step column is present and numeric
    if "step" not in df.columns or df["step"].isna().all():
        df["step"] = range(len(df))
    
    # Convert step to numeric, handling any non-numeric values
    try:
        df["step"] = pd.to_numeric(df["step"], errors='coerce')
        # Fill any NaN values with sequential numbers
        if df["step"].isna().any():
            df["step"] = df["step"].fillna(range(len(df)))
    except Exception as e:
        raise ValidationError(
            f"Failed to convert step column to numeric: {e}",
            suggestion="Ensure step values are numeric",
            error_code="INVALID_STEP_COLUMN"
        ) from e

    # Sort by step
    df = df.sort_values("step").reset_index(drop=True)

    return df[required_cols]


def _detect_adapter_type(source: Union[str, Path]) -> str:
    """Auto-detect adapter type from source content."""
    source_path = Path(source)

    # Note: WandB URIs are handled in ingest_runs() before this function is called
    # This function only handles local file/directory detection

    if not source_path.exists():
        return "trl"  # Default fallback

    # Check for WandB directory structure (more specific matching)
    # Look for wandb directory name or wandb subdirectory patterns
    # Only check for WandB if the path actually looks like a WandB structure
    if (source_path.name == "wandb" or 
        any(part == "wandb" for part in source_path.parts)):
        # Check if this looks like a WandB directory structure without instantiating adapter
        # WandB directories typically contain run-* subdirectories or specific files
        if source_path.is_dir():
            # Look for WandB-specific patterns in the directory
            wandb_indicators = [
                "run-",  # WandB run directories
                "config.yaml",  # WandB config file
                "files",  # WandB files directory
                "logs"  # WandB logs directory
            ]
            
            # Check if directory contains WandB-specific files/subdirs
            has_wandb_structure = any(
                any(item.name.startswith(indicator) for item in source_path.iterdir())
                for indicator in wandb_indicators
            )
            
            if has_wandb_structure:
                return "wandb"
        
        # Also check if this is a WandB run directory itself (starts with run-)
        if source_path.name.startswith("run-"):
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
