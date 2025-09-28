"""Main ingest function for training runs."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import pandas as pd

from ..io.event_schema import Event, dataframe_to_events
from ..utils.error_handling import (
    AdapterError,
    ValidationError,
    validate_file_path,
)
from ..utils.progress import spinner
from .training_metrics_normalizer import standardize_training_metrics


if TYPE_CHECKING:  # pragma: no cover - used for type hints only
    from ..adapters.base import BaseAdapter

logger = logging.getLogger(__name__)


def ingest_runs(
    source: Union[str, Path],
    adapter_hint: Optional[str] = None,
    field_map: Optional[Dict[str, str]] = None,
    config_file: Optional[Union[str, Path]] = None,
    validation_mode: str = "flexible",
    required_fields: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Ingest training runs from various sources.

    Args:
        source: Path to logs directory, file, or wandb:// URI
        adapter_hint: Optional hint for adapter type ('trl', 'openrlhf', 'wandb', 'custom_jsonl', 'grpo', 'flexible')
        field_map: Optional explicit mapping from canonical to actual field names
        config_file: Optional path to YAML/JSON config file with field mapping
        validation_mode: Validation strictness - 'strict', 'flexible', or 'lenient'
        required_fields: List of required canonical field names

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
            # Provide detailed suggestions for auto-detection failure
            source_analysis = _analyze_source_format(source)
            suggestions = _get_auto_detection_suggestions(source_analysis)

            raise AdapterError(
                f"Failed to auto-detect adapter type: {e}",
                suggestion=f"Could not determine the correct adapter. {suggestions}",
                error_code="ADAPTER_DETECTION_FAILED",
                details={"source_analysis": source_analysis}
            ) from e

    # Validate adapter type
    valid_adapters = ["trl", "openrlhf", "wandb", "custom_jsonl", "grpo", "flexible"]
    if adapter_hint not in valid_adapters:
        raise ValidationError(
            f"Invalid adapter type: {adapter_hint}",
            suggestion=f"Use one of: {', '.join(valid_adapters)}",
            error_code="INVALID_ADAPTER_TYPE"
        )

    # Create appropriate adapter
    try:
        adapter = _create_adapter(
            adapter_hint,
            source,
            field_map=field_map,
            config_file=config_file,
            validation_mode=validation_mode,
            required_fields=required_fields,
            valid_adapters=valid_adapters,
        )
    except Exception as e:
        raise AdapterError(
            f"Failed to create adapter: {e}",
            suggestion="Check that the adapter type is correct and dependencies are installed",
            error_code="ADAPTER_CREATION_FAILED"
        ) from e

    # Check if adapter can handle the source
    if not adapter.can_handle():
        # Get detailed format requirements for better error messages
        format_requirements = _get_adapter_format_requirements(adapter_hint)
        source_analysis = _analyze_source_format(source)

        raise AdapterError(
            f"Adapter '{adapter_hint}' cannot handle source: {source}",
            suggestion=f"Expected format: {format_requirements['description']}\n"
                      f"Found: {source_analysis['description']}\n"
                      f"Try: {format_requirements['suggestions']}",
            error_code="ADAPTER_CANNOT_HANDLE_SOURCE",
            details={
                "adapter": adapter_hint,
                "source": str(source),
                "expected_format": format_requirements,
                "source_analysis": source_analysis
            }
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
        df = standardize_training_metrics(df)
        logger.info(f"Schema standardized, {len(df)} records ready")
    except Exception as e:
        raise AdapterError(
            f"Failed to standardize schema: {e}",
            suggestion="Check that the data contains the required fields",
            error_code="SCHEMA_STANDARDIZATION_FAILED"
        ) from e

    return df
def ingest_runs_to_events(
    source: Union[str, Path],
    adapter_hint: Optional[str] = None,
    field_map: Optional[Dict[str, str]] = None,
    config_file: Optional[Union[str, Path]] = None,
    validation_mode: str = "flexible",
    required_fields: Optional[List[str]] = None
) -> List[Event]:
    """
    Ingest training runs and convert to normalized Event objects.

    Args:
        source: Path to logs directory, file, or wandb:// URI
        adapter_hint: Optional hint for adapter type ('trl', 'openrlhf', 'wandb', 'custom_jsonl', 'grpo', 'flexible')
        field_map: Optional explicit mapping from canonical to actual field names
        config_file: Optional path to YAML/JSON config file with field mapping
        validation_mode: Validation strictness - 'strict', 'flexible', or 'lenient'
        required_fields: List of required canonical field names

    Returns:
        List of Event objects with normalized training data
    """
    # Get the DataFrame first
    df = ingest_runs(
        source,
        adapter_hint,
        field_map=field_map,
        config_file=config_file,
        validation_mode=validation_mode,
        required_fields=required_fields
    )

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


def _create_adapter(
    adapter_hint: str,
    source: Union[str, Path],
    *,
    field_map: Optional[Dict[str, str]],
    config_file: Optional[Union[str, Path]],
    validation_mode: str,
    required_fields: Optional[List[str]],
    valid_adapters: List[str],
) -> "BaseAdapter":
    """Lazily construct adapters to avoid circular imports."""

    if adapter_hint == "trl":
        from ..adapters.trl import TRLAdapter

        return TRLAdapter(source)
    if adapter_hint == "openrlhf":
        from ..adapters.openrlhf import OpenRLHFAdapter

        return OpenRLHFAdapter(source)
    if adapter_hint == "wandb":
        from ..adapters.wandb import WandBAdapter

        return WandBAdapter(source)
    if adapter_hint == "custom_jsonl":
        from ..adapters.custom_jsonl import CustomJSONLAdapter

        return CustomJSONLAdapter(source)
    if adapter_hint == "grpo":
        from ..adapters.grpo import GRPOAdapter

        return GRPOAdapter(source)
    if adapter_hint == "flexible":
        from ..adapters.flexible import FlexibleDataAdapter

        return FlexibleDataAdapter(
            source,
            field_map=field_map,
            config_file=config_file,
            required_fields=required_fields or ["step", "reward"],
            validation_mode=validation_mode,
        )

    raise ValidationError(
        f"Unknown adapter type: {adapter_hint}",
        suggestion=f"Use one of: {', '.join(valid_adapters)}",
        error_code="UNKNOWN_ADAPTER_TYPE",
    )


def _detect_adapter_type(source: Union[str, Path]) -> str:
    """Auto-detect adapter type from source content."""
    source_path = Path(source)

    if not source_path.exists():
        return "flexible"  # Default to flexible adapter

    # Check for TRL-specific patterns
    from ..adapters.trl import TRLAdapter

    trl_adapter = TRLAdapter(source_path)
    if trl_adapter.can_handle():
        return "trl"

    # Check for OpenRLHF-specific patterns
    from ..adapters.openrlhf import OpenRLHFAdapter

    openrlhf_adapter = OpenRLHFAdapter(source_path)
    if openrlhf_adapter.can_handle():
        return "openrlhf"

    # Check for GRPO-specific patterns before falling back to flexible adapter
    from ..adapters.grpo import GRPOAdapter

    grpo_adapter = GRPOAdapter(source_path)
    if grpo_adapter.can_handle():
        return "grpo"

    # Check for our custom JSONL format after known formats
    from ..adapters.custom_jsonl import CustomJSONLAdapter

    custom_adapter = CustomJSONLAdapter(source_path)
    if custom_adapter.can_handle():
        return "custom_jsonl"

    # Default to flexible adapter for better field resolution
    return "flexible"


def _get_adapter_format_requirements(adapter_type: str) -> dict:
    """Get detailed format requirements for an adapter type."""
    requirements = {
        "trl": {
            "description": "TRL training logs (JSONL or .log files)",
            "file_extensions": [".jsonl", ".log"],
            "required_fields": ["step", "phase", "reward_mean", "kl_mean"],
            "optional_fields": ["entropy_mean", "clip_frac", "grad_norm", "lr", "loss", "wall_time", "seed", "run_id", "git_sha"],
            "examples": [
                '{"step": 0, "phase": "train", "reward_mean": 0.5, "kl_mean": 0.1, "entropy_mean": 0.8}',
                '{"step": 1, "phase": "train", "reward_mean": 0.6, "kl_mean": 0.12, "loss": 0.4, "lr": 0.001}'
            ],
            "suggestions": "Ensure your data contains the required fields. For JSONL files, each line should be a valid JSON object with training metrics."
        },
        "openrlhf": {
            "description": "OpenRLHF training logs (JSONL or .log files)",
            "file_extensions": [".jsonl", ".log"],
            "required_fields": ["step", "phase", "reward_mean", "kl_mean"],
            "optional_fields": ["entropy_mean", "clip_frac", "grad_norm", "lr", "loss", "wall_time", "seed", "run_id", "git_sha"],
            "examples": [
                '{"step": 0, "phase": "train", "reward_mean": 0.5, "kl_mean": 0.1, "entropy_mean": 0.8}',
                '{"step": 1, "phase": "train", "reward_mean": 0.6, "kl_mean": 0.12, "loss": 0.4, "lr": 0.001}'
            ],
            "suggestions": "Ensure your data contains the required fields. For JSONL files, each line should be a valid JSON object with training metrics."
        },
        "custom_jsonl": {
            "description": "Custom JSONL format with specific field names",
            "file_extensions": [".jsonl"],
            "required_fields": ["global_step", "reward_scalar", "kl_to_ref"],
            "optional_fields": ["entropy", "clip_frac", "grad_norm", "lr", "loss", "wall_time", "seed", "run_id", "git_sha"],
            "examples": [
                '{"global_step": 0, "reward_scalar": 0.5, "kl_to_ref": 0.1, "entropy": 0.8}',
                '{"global_step": 1, "reward_scalar": 0.6, "kl_to_ref": 0.12, "loss": 0.4, "lr": 0.001}'
            ],
            "suggestions": "Use field names like 'global_step', 'reward_scalar', 'kl_to_ref' instead of standard names."
        },
        "wandb": {
            "description": "WandB run URI (wandb://entity/project/run_id)",
            "file_extensions": [],
            "required_fields": ["entity", "project", "run_id"],
            "optional_fields": [],
            "examples": [
                "wandb://my-entity/my-project/abc123",
                "wandb://team/project/run-2024-01-01-12-00-00"
            ],
            "suggestions": "Use the format: wandb://entity/project/run_id. Ensure you have WandB access and the run exists."
        },
        "grpo": {
            "description": "GRPO training logs (run.jsonl files with GRPO metrics)",
            "file_extensions": [".jsonl"],
            "required_fields": ["step", "reward_mean", "kl", "entropy"],
            "optional_fields": [
                "reward_std",
                "advantage_mean",
                "advantage_std",
                "grad_norm_policy",
                "grad_norm_value",
                "kl_coef",
                "accept_rate",
                "length",
            ],
            "examples": [
                '{"step": 0, "reward_mean": 0.5, "kl": 0.1, "entropy": 0.8, "advantage_mean": 0.2}',
                '{"step": 1, "reward_mean": 0.6, "kl": 0.12, "entropy": 0.82, "grad_norm_policy": 1.5}',
            ],
            "suggestions": (
                "Provide run.jsonl files emitted by GRPO training. Each line should be a JSON object with GRPO metrics, "
                "including reward_mean, kl, entropy, and optional extras like advantage statistics."
            ),
        },
        "flexible": {
            "description": "Flexible adapter supporting multiple formats with automatic field resolution",
            "file_extensions": [".jsonl", ".json", ".csv", ".parquet"],
            "required_fields": ["step", "reward"],
            "optional_fields": ["kl", "entropy", "loss", "phase", "wall_time", "seed", "run_id", "git_sha", "lr", "grad_norm", "clip_frac", "tokens_in", "tokens_out"],
            "examples": [
                '{"step": 0, "reward": 0.5, "kl": 0.1, "entropy": 0.8}',
                '{"global_step": 0, "reward_scalar": 0.5, "kl_to_ref": 0.1, "entropy": 0.8}',
                '{"iteration": 0, "score": 0.5, "kl_divergence": 0.1, "policy_entropy": 0.8}'
            ],
            "suggestions": "Supports automatic field resolution for common field names. Use field_map for custom schemas. Supports nested fields with dot notation."
        }
    }

    return requirements.get(adapter_type, {
        "description": f"Unknown adapter type: {adapter_type}",
        "file_extensions": [],
        "required_fields": [],
        "optional_fields": [],
        "examples": [],
        "suggestions": "Use a valid adapter type: trl, openrlhf, custom_jsonl, wandb"
    })


def _analyze_source_format(source: Union[str, Path]) -> dict:
    """Analyze the source format and provide detailed information."""
    source_path = Path(source)

    if not source_path.exists():
        return {
            "description": "Source does not exist",
            "type": "missing",
            "files": [],
            "issues": ["Path does not exist"]
        }

    if source_path.is_file():
        return _analyze_file_format(source_path)
    elif source_path.is_dir():
        return _analyze_directory_format(source_path)
    else:
        return {
            "description": "Unknown source type",
            "type": "unknown",
            "files": [],
            "issues": ["Source is neither file nor directory"]
        }


def _analyze_file_format(file_path: Path) -> dict:
    """Analyze a single file format."""
    issues = []
    file_type = "unknown"

    if file_path.suffix == ".jsonl":
        file_type = "jsonl"
        # Try to read first few lines to analyze content
        try:
            with open(file_path) as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= 3:  # Only read first 3 lines
                        break
                    lines.append(line.strip())

                if lines:
                    # Try to parse JSON
                    import json
                    sample_data = []
                    for line_num, line in enumerate(lines, 1):  # Start line numbering from 1
                        if line:
                            try:
                                data = json.loads(line)
                                sample_data.append(data)
                            except json.JSONDecodeError:
                                issues.append(f"Invalid JSON on line {line_num}")

                    if sample_data:
                        # Analyze field structure
                        all_fields = set()
                        for data in sample_data:
                            if isinstance(data, dict):
                                all_fields.update(data.keys())

                        required_fields = ["step", "phase", "reward_mean", "kl_mean"]
                        missing_fields = [f for f in required_fields if f not in all_fields]

                        if missing_fields:
                            issues.append(f"Missing required fields: {missing_fields}")

                        # Check for custom format indicators
                        custom_indicators = ["global_step", "reward_scalar", "kl_to_ref"]
                        has_custom = any(f in all_fields for f in custom_indicators)

                        if has_custom:
                            issues.append("Contains custom format fields - try 'custom_jsonl' adapter")

                        return {
                            "description": f"JSONL file with {len(sample_data)} sample records",
                            "type": "jsonl",
                            "files": [str(file_path)],
                            "fields_found": list(all_fields),
                            "issues": issues,
                            "sample_data": sample_data[0] if sample_data else None
                        }
                    else:
                        issues.append("No valid JSON records found")
                else:
                    issues.append("File is empty")
        except Exception as e:
            issues.append(f"Error reading file: {e}")

    elif file_path.suffix == ".log":
        file_type = "log"
        try:
            with open(file_path) as f:
                content = f.read(1000)  # Read first 1000 chars
                if not content.strip():
                    issues.append("Log file is empty")
                else:
                    # Check for common patterns
                    if "trl" in content.lower():
                        issues.append("Contains TRL keywords - try 'trl' adapter")
                    elif "openrlhf" in content.lower() or "rlhf" in content.lower():
                        issues.append("Contains OpenRLHF keywords - try 'openrlhf' adapter")
        except Exception as e:
            issues.append(f"Error reading log file: {e}")

    else:
        issues.append(f"Unsupported file extension: {file_path.suffix}")

    return {
        "description": f"{file_type} file: {file_path.name}",
        "type": file_type,
        "files": [str(file_path)],
        "issues": issues
    }


def _analyze_directory_format(dir_path: Path) -> dict:
    """Analyze a directory format."""
    issues = []
    files_found = []

    # Look for common log files
    jsonl_files = list(dir_path.glob("*.jsonl"))
    log_files = list(dir_path.glob("*.log"))

    files_found = jsonl_files + log_files

    if not files_found:
        issues.append("No .jsonl or .log files found in directory")
        return {
            "description": "Directory with no log files",
            "type": "directory",
            "files": [],
            "issues": issues
        }

    # Analyze the first few files
    sample_issues = []
    for file_path in files_found[:3]:  # Analyze first 3 files
        file_analysis = _analyze_file_format(file_path)
        if file_analysis.get("issues"):
            sample_issues.extend(file_analysis["issues"])

    issues.extend(sample_issues)

    return {
        "description": f"Directory with {len(files_found)} log files ({len(jsonl_files)} JSONL, {len(log_files)} .log)",
        "type": "directory",
        "files": [str(f) for f in files_found],
        "issues": issues
    }


def _get_auto_detection_suggestions(source_analysis: dict) -> str:
    """Get suggestions for adapter selection based on source analysis."""
    suggestions = []

    if source_analysis["type"] == "missing":
        suggestions.append("Ensure the source path exists and is accessible.")
        return " ".join(suggestions)

    if source_analysis["type"] == "jsonl":
        fields_found = source_analysis.get("fields_found", [])

        # Check for custom format indicators
        custom_indicators = ["global_step", "reward_scalar", "kl_to_ref"]
        has_custom = any(f in fields_found for f in custom_indicators)

        if has_custom:
            suggestions.append("Your data contains custom field names (global_step, reward_scalar, kl_to_ref). Try: --adapter custom_jsonl")
        else:
            # Check for standard format
            standard_fields = ["step", "phase", "reward_mean", "kl_mean"]
            has_standard = any(f in fields_found for f in standard_fields)

            if has_standard:
                suggestions.append("Your data appears to be in standard format. Try: --adapter trl or --adapter openrlhf")
            else:
                suggestions.append("Your data doesn't match standard formats. Try: --adapter custom_jsonl")

    elif source_analysis["type"] == "log":
        suggestions.append("You have .log files. Try: --adapter trl or --adapter openrlhf")

    elif source_analysis["type"] == "directory":
        files = source_analysis.get("files", [])
        if any(f.endswith('.jsonl') for f in files):
            suggestions.append("Directory contains JSONL files. Try: --adapter trl, --adapter openrlhf, or --adapter custom_jsonl")
        elif any(f.endswith('.log') for f in files):
            suggestions.append("Directory contains .log files. Try: --adapter trl or --adapter openrlhf")
        else:
            suggestions.append("Directory doesn't contain recognized log files.")

    # Add general suggestions
    suggestions.append("Available adapters: trl, openrlhf, custom_jsonl, wandb")
    suggestions.append("Use --adapter <type> to specify the adapter explicitly.")

    return " ".join(suggestions)
