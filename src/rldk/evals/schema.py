"""Schema definitions and validation for evaluation inputs."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
import logging

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
        if "output" in missing_required:
            error_msg += ". Provide one of: output, response, completion, text"
        elif "step" in missing_required:
            error_msg += ". Provide one of: step, global_step, iteration, epoch"
        raise ValueError(error_msg)
    
    # Step 3: Check for missing optional columns and add warnings
    missing_optional = []
    for col_spec in schema.optional_columns:
        if col_spec.name not in data.columns:
            missing_optional.append(col_spec.name)
    
    if missing_optional:
        if "events" in missing_optional:
            warnings.append(f"events column not provided, event-based diagnostics will be skipped")
        else:
            warnings.append(f"Optional columns not provided: {', '.join(missing_optional)}")
    
    # Step 4: Basic dtype validation (where reasonable)
    for col_spec in schema.get_all_columns():
        if col_spec.name in data.columns:
            try:
                if col_spec.dtype == "numeric":
                    # Try to convert to numeric, keeping original if fails
                    pd.to_numeric(data[col_spec.name], errors='raise')
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


def safe_mean(values: List[float]) -> Optional[float]:
    """
    Calculate mean of values, returning None if empty or all NaN.
    
    Args:
        values: List of numeric values
        
    Returns:
        Mean value or None if no valid values
    """
    if not values:
        return None
    
    # Filter out NaN values
    valid_values = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
    
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
    # For now, all suites use the standard schema
    # In the future, we could customize schemas per suite
    return STANDARD_EVAL_SCHEMA