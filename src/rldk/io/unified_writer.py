"""Unified writer interface with consistent error handling and file naming conventions."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RLDebugKitIOError(Exception):
    """Base exception for RL Debug Kit IO operations."""
    pass


class FileWriteError(RLDebugKitIOError):
    """Exception raised when file writing fails."""
    pass


class SchemaValidationError(RLDebugKitIOError):
    """Exception raised when schema validation fails."""
    pass


class UnifiedWriter:
    """Unified writer interface with consistent error handling and file naming conventions."""

    def __init__(self, output_dir: Union[str, Path], create_dirs: bool = True):
        """
        Initialize unified writer.

        Args:
            output_dir: Base output directory for all files
            create_dirs: Whether to create directories automatically
        """
        self.output_dir = Path(output_dir)
        if create_dirs:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_directory(self, file_path: Path) -> None:
        """Ensure parent directory exists for file path."""
        file_path.parent.mkdir(parents=True, exist_ok=True)

    def _validate_jsonl_data(self, data: List[Dict[str, Any]]) -> List[str]:
        """
        Validate JSONL data for consistency and quality.

        Args:
            data: List of dictionaries to validate

        Returns:
            List of validation warnings/errors
        """
        warnings = []

        if not data:
            warnings.append("Empty data list provided")
            return warnings

        # Check for consistent field structure
        all_keys = set()
        for item in data:
            all_keys.update(item.keys())

        # Check for missing fields across records
        for key in all_keys:
            missing_count = sum(1 for item in data if key not in item)
            if missing_count > 0:
                warnings.append(f"Field '{key}' missing in {missing_count}/{len(data)} records")

        # Check for NaN values that should be None
        for i, item in enumerate(data):
            for key, value in item.items():
                if pd.isna(value) and value is not None:
                    warnings.append(f"Record {i}: Field '{key}' contains pandas NaN instead of None")
                elif isinstance(value, (float, np.floating)) and np.isnan(value):
                    warnings.append(f"Record {i}: Field '{key}' contains numpy NaN instead of None")

        return warnings

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for numpy types and special objects with consistent NaN handling."""
        # Handle numpy scalars first
        if hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        # Handle numpy arrays
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # Handle NaN values consistently - convert to None
        elif isinstance(obj, (float, np.floating)) and np.isnan(obj):
            return None
        # Handle infinity values consistently - convert to None
        elif isinstance(obj, (float, np.floating)) and np.isinf(obj):
            return None
        # Handle datetime objects
        elif isinstance(obj, datetime):
            return obj.isoformat()
        # Handle pandas NaN values
        elif pd.isna(obj):
            return None
        # Handle nested structures recursively
        elif isinstance(obj, dict):
            return {key: self._json_serializer(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._json_serializer(item) for item in obj]
        else:
            return obj

    def write_json(
        self,
        data: Dict[str, Any],
        filename: str,
        validate_schema: Optional[Callable] = None,
        indent: int = 2
    ) -> Path:
        """
        Write data to JSON file with consistent error handling.

        Args:
            data: Dictionary to write
            filename: Output filename (will be placed in output_dir)
            validate_schema: Optional schema validation function
            indent: JSON indentation level

        Returns:
            Path to written file

        Raises:
            FileWriteError: If file writing fails
            SchemaValidationError: If schema validation fails
        """
        try:
            # Validate schema if provided
            if validate_schema:
                try:
                    validate_schema(data)
                except Exception as e:
                    raise SchemaValidationError(f"Schema validation failed: {e}")

            # Prepare file path
            file_path = self.output_dir / filename
            self._ensure_directory(file_path)

            # Write JSON with custom serializer
            with open(file_path, "w") as f:
                json.dump(data, f, indent=indent, default=self._json_serializer)

            logger.info(f"Successfully wrote JSON file: {file_path}")
            return file_path

        except Exception as e:
            error_msg = f"Failed to write JSON file {filename}: {e}"
            logger.error(error_msg)
            raise FileWriteError(error_msg) from e

    def write_jsonl(
        self,
        data: List[Dict[str, Any]],
        filename: str,
        validate_schema: Optional[Callable] = None
    ) -> Path:
        """
        Write data to JSONL file with consistent error handling and data validation.

        Args:
            data: List of dictionaries to write
            filename: Output filename
            validate_schema: Optional schema validation function

        Returns:
            Path to written file
        """
        try:
            # Validate data consistency
            warnings = self._validate_jsonl_data(data)
            if warnings:
                logger.warning(f"Data validation warnings for {filename}: {warnings}")

            # Validate schema if provided
            if validate_schema:
                for i, item in enumerate(data):
                    try:
                        validate_schema(item)
                    except Exception as e:
                        raise SchemaValidationError(f"Schema validation failed for item {i}: {e}")

            # Prepare file path
            file_path = self.output_dir / filename
            self._ensure_directory(file_path)

            # Write JSONL with consistent serialization
            with open(file_path, "w") as f:
                for item in data:
                    # Ensure consistent serialization of each item
                    serialized_item = self._json_serializer(item)
                    json.dump(serialized_item, f, default=self._json_serializer)
                    f.write("\n")

            logger.info(f"Successfully wrote JSONL file: {file_path}")
            return file_path

        except Exception as e:
            error_msg = f"Failed to write JSONL file {filename}: {e}"
            logger.error(error_msg)
            raise FileWriteError(error_msg) from e

    def write_csv(
        self,
        df: pd.DataFrame,
        filename: str,
        index: bool = False
    ) -> Path:
        """
        Write DataFrame to CSV file with consistent error handling.

        Args:
            df: DataFrame to write
            filename: Output filename
            index: Whether to include DataFrame index

        Returns:
            Path to written file
        """
        try:
            file_path = self.output_dir / filename
            self._ensure_directory(file_path)

            df.to_csv(file_path, index=index)

            logger.info(f"Successfully wrote CSV file: {file_path}")
            return file_path

        except Exception as e:
            error_msg = f"Failed to write CSV file {filename}: {e}"
            logger.error(error_msg)
            raise FileWriteError(error_msg) from e

    def write_markdown(
        self,
        content: str,
        filename: str
    ) -> Path:
        """
        Write content to markdown file with consistent error handling.

        Args:
            content: Markdown content to write
            filename: Output filename

        Returns:
            Path to written file
        """
        try:
            file_path = self.output_dir / filename
            self._ensure_directory(file_path)

            with open(file_path, "w") as f:
                f.write(content)

            logger.info(f"Successfully wrote markdown file: {file_path}")
            return file_path

        except Exception as e:
            error_msg = f"Failed to write markdown file {filename}: {e}"
            logger.error(error_msg)
            raise FileWriteError(error_msg) from e

    def write_png(
        self,
        fig: plt.Figure,
        filename: str,
        dpi: int = 150,
        bbox_inches: str = "tight"
    ) -> Path:
        """
        Save matplotlib figure as PNG with consistent error handling.

        Args:
            fig: Matplotlib figure to save
            filename: Output filename
            dpi: DPI for the saved image
            bbox_inches: Bbox parameter for tight layout

        Returns:
            Path to written file
        """
        try:
            file_path = self.output_dir / filename
            self._ensure_directory(file_path)

            fig.savefig(file_path, dpi=dpi, bbox_inches=bbox_inches)

            logger.info(f"Successfully wrote PNG file: {file_path}")
            return file_path

        except Exception as e:
            error_msg = f"Failed to write PNG file {filename}: {e}"
            logger.error(error_msg)
            raise FileWriteError(error_msg) from e

    def write_metrics_jsonl(
        self,
        df: pd.DataFrame,
        filename: str,
        schema_class: Optional[Any] = None
    ) -> Path:
        """
        Write metrics DataFrame to JSONL file with consistent schema validation and formatting.

        Args:
            df: DataFrame with training metrics
            filename: Output filename
            schema_class: Optional Pydantic schema class for validation

        Returns:
            Path to written file
        """
        try:
            file_path = self.output_dir / filename
            self._ensure_directory(file_path)

            # Always use schema-based approach for consistency
            if schema_class:
                # Convert DataFrame to schema objects with proper NaN handling
                schema = schema_class.from_dataframe(df)
                with open(file_path, "w") as f:
                    for metric in schema.metrics:
                        # Use model_dump with consistent serialization
                        metric_dict = metric.model_dump()
                        json.dump(metric_dict, f, default=self._json_serializer)
                        f.write("\n")
            else:
                # Fallback: Write directly from DataFrame with consistent NaN handling
                with open(file_path, "w") as f:
                    for _, row in df.iterrows():
                        # Convert row to dict and handle NaN values consistently
                        row_dict = row.to_dict()
                        # Ensure all NaN values are converted to None
                        for key, value in row_dict.items():
                            if pd.isna(value):
                                row_dict[key] = None
                        json.dump(row_dict, f, default=self._json_serializer)
                        f.write("\n")

            logger.info(f"Successfully wrote metrics JSONL file: {file_path}")
            return file_path

        except Exception as e:
            error_msg = f"Failed to write metrics JSONL file {filename}: {e}"
            logger.error(error_msg)
            raise FileWriteError(error_msg) from e


# Convenience functions for backward compatibility
def write_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Write JSON data to file (backward compatibility)."""
    path = Path(path)
    writer = UnifiedWriter(path.parent)
    writer.write_json(data, path.name)


def write_png(fig: plt.Figure, path: Union[str, Path]) -> None:
    """Write PNG figure to file (backward compatibility)."""
    path = Path(path)
    writer = UnifiedWriter(path.parent)
    writer.write_png(fig, path.name)


def mkdir_reports() -> Path:
    """Create rldk_reports directory and return path (backward compatibility)."""
    reports_dir = Path("rldk_reports")
    reports_dir.mkdir(exist_ok=True)
    return reports_dir


def write_metrics_jsonl(df: pd.DataFrame, file_path: Union[str, Path]) -> None:
    """Write metrics DataFrame to JSONL file (backward compatibility)."""
    file_path = Path(file_path)
    writer = UnifiedWriter(file_path.parent)
    writer.write_metrics_jsonl(df, file_path.name)
