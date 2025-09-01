"""Base adapter class for training log formats."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union
import pandas as pd


class BaseAdapter(ABC):
    """Base class for training log adapters."""

    def __init__(self, source: Union[str, Path]):
        self.source = Path(source) if isinstance(source, str) else source

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
        return True
