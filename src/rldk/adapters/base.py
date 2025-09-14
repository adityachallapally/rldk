"""Base adapter class for training log formats."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from ..utils.error_handling import AdapterError, ValidationError


class BaseAdapter(ABC):
    """Base class for training log adapters."""

    def __init__(self, source: Union[str, Path]):
        self.source = Path(source) if isinstance(source, str) else source
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def can_handle(self) -> bool:
        """Check if this adapter can handle the given source."""
        pass

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load and convert training logs to standard format."""
        pass

    def get_metadata(self) -> dict:
        """Get metadata about the training run."""
        return {}

    def validate(self) -> bool:
        """Validate that the source contains expected data."""
        try:
            return self._validate_source()
        except Exception as e:
            self.logger.warning(f"Validation failed: {e}")
            return False

    def _validate_source(self) -> bool:
        """Internal validation method that can be overridden by subclasses."""
        if not self.source.exists():
            raise ValidationError(
                f"Source does not exist: {self.source}",
                suggestion="Check that the source path is correct",
                error_code="SOURCE_NOT_FOUND"
            )
        return True

    def _safe_load(self) -> pd.DataFrame:
        """Safely load data with error handling."""
        try:
            return self.load()
        except Exception as e:
            raise AdapterError(
                f"Failed to load data from {self.source}: {e}",
                suggestion="Check that the source contains valid data in the expected format",
                error_code="LOAD_FAILED",
                details={"source": str(self.source), "adapter": self.__class__.__name__}
            ) from e

    def _validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the loaded DataFrame."""
        if df.empty:
            raise AdapterError(
                "No data found in source",
                suggestion="Check that the source contains valid training data",
                error_code="NO_DATA_FOUND"
            )

        # Check for required columns
        required_columns = ["step"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            self.logger.warning(f"Missing required columns: {missing_columns}")
            # Add missing columns with default values
            for col in missing_columns:
                if col == "step":
                    df[col] = range(len(df))

        return df

    def _handle_file_error(self, file_path: Path, operation: str) -> None:
        """Handle file operation errors with helpful messages."""
        if not file_path.exists():
            raise AdapterError(
                f"File not found: {file_path}",
                suggestion="Check that the file path is correct and the file exists",
                error_code="FILE_NOT_FOUND"
            )

        if not file_path.is_file():
            raise AdapterError(
                f"Path is not a file: {file_path}",
                suggestion="Ensure the path points to a file, not a directory",
                error_code="NOT_A_FILE"
            )

        try:
            with open(file_path) as f:
                f.read(1)  # Try to read one character
        except PermissionError:
            raise AdapterError(
                f"Permission denied: {file_path}",
                suggestion="Check file permissions and ensure you have read access",
                error_code="PERMISSION_DENIED"
            )
        except UnicodeDecodeError as e:
            raise AdapterError(
                f"File encoding error: {file_path}",
                suggestion="Check that the file is in a supported encoding (UTF-8)",
                error_code="ENCODING_ERROR",
                details={"error": str(e)}
            )
        except Exception as e:
            raise AdapterError(
                f"File {operation} failed: {file_path}",
                suggestion="Check that the file is accessible and not corrupted",
                error_code="FILE_ACCESS_ERROR",
                details={"error": str(e)}
            ) from e

    def _log_operation(self, operation: str, details: Optional[Dict[str, Any]] = None):
        """Log adapter operations for debugging."""
        message = f"{self.__class__.__name__}: {operation}"
        if details:
            message += f" - {details}"
        self.logger.debug(message)
