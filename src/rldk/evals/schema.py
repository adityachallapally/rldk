"""Schema definitions and validation for evaluation inputs."""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from ..utils.error_handling import ValidationError

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ColumnSpec:
    """Specification for a data column in evaluation inputs."""

    name: str
    dtype: str
    required: bool
    description: str
    example: Any
    synonyms: Optional[List[str]] = None

    def __post_init__(self):
        if self.synonyms is None:
            self.synonyms = []


@dataclass
class EvalInputSchema:
    """Schema definition for evaluation input data."""

    required_columns: List[ColumnSpec]
    optional_columns: List[ColumnSpec]

    def get_all_columns(self) -> List[ColumnSpec]:
        """Get all columns (required + optional)."""
        return self.required_columns + self.optional_columns

    def get_column_by_name(self, name: str) -> Optional[ColumnSpec]:
        """Get column specification by name or synonym."""
        for col in self.get_all_columns():
            if col.name == name:
                return col
            if col.synonyms and name in col.synonyms:
                return col
        return None

    def get_column_names(self) -> List[str]:
        """Get all column names and synonyms."""
        names = []
        for col in self.get_all_columns():
            names.append(col.name)
            if col.synonyms:
                names.extend(col.synonyms)
        return names


@dataclass
class ValidatedFrame:
    """Result of input validation with normalized DataFrame and metadata."""

    data: pd.DataFrame
    warnings: List[str]
    errors: List[str]
    normalized_columns: Dict[str, str]  # original_name -> normalized_name


# Define the standard evaluation input schema
STANDARD_EVAL_SCHEMA = EvalInputSchema(
    required_columns=[
        ColumnSpec(
            name="step",
            dtype="numeric",
            required=True,
            description="Training step or global step number for temporal analysis",
            example=1000,
            synonyms=["global_step", "iteration", "epoch"]
        ),
        ColumnSpec(
            name="output",
            dtype="text",
            required=True,
            description="Model output text for evaluation",
            example="This is a helpful response.",
            synonyms=["response", "completion", "text", "generation"]
        )
    ],
    optional_columns=[
        ColumnSpec(
            name="reward",
            dtype="numeric",
            required=False,
            description="Reward signal for the output",
            example=0.85,
            synonyms=["reward_mean", "score", "value"]
        ),
        ColumnSpec(
            name="kl_to_ref",
            dtype="numeric",
            required=False,
            description="KL divergence to reference model",
            example=0.12,
            synonyms=["kl", "kl_divergence", "kl_mean"]
        ),
        ColumnSpec(
            name="events",
            dtype="object",
            required=False,
            description="Event logs for detailed analysis",
            example=[{"event_type": "token_generated", "timestamp": "2024-01-01T00:00:00Z", "token_count": 10}],
            synonyms=["event_logs", "logs", "events_raw"]
        )
    ]
)


# Define the RL metrics-only evaluation input schema
RL_METRICS_SCHEMA = EvalInputSchema(
    required_columns=[
        ColumnSpec(
            name="step",
            dtype="numeric",
            required=True,
            description="Training step or global step number for temporal analysis",
            example=1000,
            synonyms=["global_step", "iteration", "epoch"]
        )
    ],
    optional_columns=[
        ColumnSpec(
            name="reward",
            dtype="numeric",
            required=False,
            description="Reward signal for the output",
            example=0.85,
            synonyms=["reward_mean", "score"]
        ),
        ColumnSpec(
            name="kl",
            dtype="numeric",
            required=False,
            description="KL divergence to reference model",
            example=0.12,
            synonyms=["kl_mean", "kl_to_ref"]
        ),
        ColumnSpec(
            name="entropy",
            dtype="numeric",
            required=False,
            description="Entropy of the model outputs",
            example=5.2,
            synonyms=["entropy_mean"]
        ),
        ColumnSpec(
            name="tokens_in",
            dtype="numeric",
            required=False,
            description="Number of input tokens processed",
            example=512,
            synonyms=["input_tokens", "prompt_tokens"]
        ),
        ColumnSpec(
            name="tokens_out",
            dtype="numeric",
            required=False,
            description="Number of output tokens generated",
            example=128,
            synonyms=["output_tokens", "generated_tokens"]
        ),
        ColumnSpec(
            name="episode_id",
            dtype="numeric",
            required=False,
            description="Episode identifier for RL training",
            example=42,
            synonyms=["ep_id", "episode"]
        ),
        ColumnSpec(
            name="episode_return",
            dtype="numeric",
            required=False,
            description="Total return for the episode",
            example=15.7,
            synonyms=["ep_return", "total_return"]
        ),
        ColumnSpec(
            name="episode_length",
            dtype="numeric",
            required=False,
            description="Length of the episode in steps",
            example=200,
            synonyms=["ep_length", "episode_steps"]
        )
    ]
)


def normalize_columns(df: pd.DataFrame, column_mapping: Optional[Dict[str, str]] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Apply user column mapping first, then synonym normalization.

    Args:
        df: Input DataFrame
        column_mapping: User-provided mapping from original to target column names

    Returns:
        Tuple of (normalized_dataframe, effective_column_mapping)
    """
    data = df.copy()
    effective_mapping = {}

    # Step 1: Apply user-provided column mapping first
    if column_mapping:
        for original_col, target_col in column_mapping.items():
            if original_col in data.columns:
                data = data.rename(columns={original_col: target_col})
                effective_mapping[original_col] = target_col
                logger.debug(f"User mapping: '{original_col}' -> '{target_col}'")

    # Step 2: Apply synonym normalization for remaining columns
    all_schemas = [STANDARD_EVAL_SCHEMA, RL_METRICS_SCHEMA]

    for schema in all_schemas:
        for col_spec in schema.get_all_columns():
            for synonym in col_spec.synonyms:
                if synonym in data.columns and col_spec.name not in data.columns:
                    data = data.rename(columns={synonym: col_spec.name})
                    logger.debug(f"Synonym mapping: '{synonym}' -> '{col_spec.name}'")

    return data, effective_mapping


def validate_eval_input(
    df: pd.DataFrame,
    schema: EvalInputSchema = STANDARD_EVAL_SCHEMA,
    suite_name: str = "generic"
) -> ValidatedFrame:
    """
    Validate and normalize evaluation input DataFrame.

    Args:
        df: Input DataFrame to validate
        schema: Schema to validate against
        suite_name: Name of the evaluation suite for context

    Returns:
        ValidatedFrame with normalized data, warnings, and errors

    Raises:
        ValueError: If required columns are missing after normalization
    """
    warnings = []
    errors = []
    normalized_columns = {}

    # Create a copy to avoid modifying the original
    data = df.copy()

    # Step 1: Column normalization
    for col_spec in schema.get_all_columns():
        # Check for exact match first
        if col_spec.name in data.columns:
            continue

        # Check for synonyms
        found_synonym = None
        for synonym in col_spec.synonyms:
            if synonym in data.columns:
                found_synonym = synonym
                break

        if found_synonym:
            # Rename column to standard name
            data = data.rename(columns={found_synonym: col_spec.name})
            normalized_columns[found_synonym] = col_spec.name
            logger.debug(f"Normalized column '{found_synonym}' to '{col_spec.name}'")

    # Step 2: Check for missing required columns
    missing_required = []
    for col_spec in schema.required_columns:
        if col_spec.name not in data.columns:
            # Build helpful error message with synonyms
            synonyms_str = ", ".join(col_spec.synonyms) if col_spec.synonyms else "none"
            missing_required.append(f"{col_spec.name} (synonyms: {synonyms_str})")

    if missing_required:
        error_msg = f"Missing required columns: {', '.join(missing_required)}"
        missing_cols = [col_spec.name for col_spec in schema.required_columns if col_spec.name not in data.columns]
        if "output" in missing_cols:
            error_msg += ". Provide one of: output, response, completion, text"
        elif "step" in missing_cols:
            error_msg += ". Provide one of: step, global_step, iteration, epoch"
        raise ValueError(error_msg)

    # Step 3: Check for missing optional columns and add warnings
    missing_optional = []
    for col_spec in schema.optional_columns:
        if col_spec.name not in data.columns:
            missing_optional.append(col_spec.name)

    if missing_optional:
        if "events" in missing_optional:
            warnings.append("events column not provided, event-based diagnostics will be skipped")
        else:
            warnings.append(f"Optional columns not provided: {', '.join(missing_optional)}")

    # Step 4: Basic dtype validation (where reasonable)
    for col_spec in schema.get_all_columns():
        if col_spec.name in data.columns:
            try:
                if col_spec.dtype == "numeric":
                    # Try to convert to numeric, keeping original if fails
                    data[col_spec.name] = pd.to_numeric(data[col_spec.name], errors='raise')
                elif col_spec.dtype == "text":
                    # Ensure it's string-like
                    data[col_spec.name] = data[col_spec.name].astype(str)
                elif col_spec.dtype == "object":
                    # Keep as-is for complex objects
                    pass
            except (ValueError, TypeError) as e:
                warnings.append(f"Column '{col_spec.name}' may not be valid {col_spec.dtype}: {e}")

    # Step 5: Check for empty DataFrame
    if len(data) == 0:
        warnings.append("DataFrame is empty")

    # Step 6: Check for all-NaN columns
    for col in data.columns:
        if data[col].isna().all():
            warnings.append(f"Column '{col}' contains only NaN values")

    return ValidatedFrame(
        data=data,
        warnings=warnings,
        errors=errors,
        normalized_columns=normalized_columns
    )


def safe_mean(values: List[Any]) -> Optional[float]:
    """
    Calculate mean of values, handling NaN and None gracefully.

    Args:
        values: List of numeric values that may contain NaN or None

    Returns:
        Mean of valid values, or None if no valid values
    """
    if not values:
        return None

    # Filter out None and NaN values
    valid_values = []
    for v in values:
        if v is not None:
            try:
                float_val = float(v)
                if not np.isnan(float_val):
                    valid_values.append(float_val)
            except (ValueError, TypeError):
                # Skip non-numeric values
                continue

    if not valid_values:
        return None

    return float(np.mean(valid_values))


def get_schema_for_suite(suite_name: str) -> EvalInputSchema:
    """
    Get schema for a specific evaluation suite.

    Args:
        suite_name: Name of the evaluation suite

    Returns:
        EvalInputSchema for the suite
    """
    if suite_name == "training_metrics":
        return RL_METRICS_SCHEMA
    else:
        # For all other suites, use the standard schema
        return STANDARD_EVAL_SCHEMA


def _create_enhanced_validation_error(
    original_error: str,
    df: pd.DataFrame,
    schema: EvalInputSchema,
    suite_name: str,
    column_mapping: Dict[str, str]
) -> 'ValidationError':
    """Create enhanced validation error with column mapping suggestions and format examples."""
    from ..adapters.field_resolver import FieldResolver
    from ..utils.error_handling import ValidationError

    available_columns = df.columns.tolist()
    field_resolver = FieldResolver()

    # Find missing required columns (check both raw and after any existing mapping)
    missing_required = []
    for col_spec in schema.required_columns:
        column_exists = (col_spec.name in available_columns or
                        col_spec.name in column_mapping.values())
        if not column_exists:
            missing_required.append(col_spec.name)

    suggestions = []
    field_map_suggestion = {}

    for missing_field in missing_required:
        field_suggestions = field_resolver.get_suggestions(missing_field, available_columns)
        if field_suggestions:
            suggestions.append(f"  {missing_field}: try {', '.join(field_suggestions)}")
            field_map_suggestion[field_suggestions[0]] = missing_field

    # Create detailed error message
    error_msg = f"Missing required columns for {suite_name} evaluation: {', '.join(missing_required)}"

    if suggestions:
        error_msg += "\n\nColumn mapping suggestions:\n" + "\n".join(suggestions)

    if field_map_suggestion:
        field_map_str = ", ".join([f'"{k}": "{v}"' for k, v in field_map_suggestion.items()])
        error_msg += f"\n\nTry this column_mapping: {{{field_map_str}}}"

    error_msg += f"\n\nSupported data formats for {suite_name}:"
    if suite_name == "training_metrics":
        error_msg += "\n- RL metrics format: step, reward, kl, entropy, loss"
        error_msg += "\n- Example: {'step': 100, 'reward': 0.5, 'kl': 0.1}"
    else:
        error_msg += "\n- Standard format: step, output (required)"
        error_msg += "\n- Example: {'step': 100, 'output': 'model response text'}"

    error_msg += f"\n\nAvailable columns in your data: {', '.join(available_columns)}"

    suggestion = "Check column names and provide column_mapping parameter if needed"
    if field_map_suggestion:
        suggestion = f"Use column_mapping parameter: {field_map_suggestion}"

    return ValidationError(
        error_msg,
        suggestion=suggestion,
        error_code="MISSING_REQUIRED_COLUMNS",
        details={
            "missing_columns": missing_required,
            "available_columns": available_columns,
            "suggested_mapping": field_map_suggestion,
            "suite_name": suite_name
        }
    )
