"""Configuration settings for RL Debug Kit."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MemorySettings:
    """Memory management settings."""
    max_parameter_size_mb: int = 50
    max_total_memory_mb: int = 500
    cleanup_threshold_mb: int = 100


@dataclass
class FileSettings:
    """File operation settings."""
    max_file_size_mb: int = 5
    max_read_size_kb: int = 512
    max_files_to_process: int = 50
    encoding: str = "utf-8"


@dataclass
class StatisticalSettings:
    """Statistical calculation settings."""
    default_confidence_level: float = 0.95
    conservative_std_estimate: float = 0.3
    binomial_threshold: float = 0.1
    cv_threshold: float = 0.5


@dataclass
class IntegritySettings:
    """Integrity check settings."""
    duplicate_threshold: float = 0.1
    pattern_threshold: float = 0.3
    leakage_threshold: float = 0.05
    max_penalty_per_check: float = 0.3


@dataclass
class RLDebugKitSettings:
    """Main configuration class."""
    memory: MemorySettings = MemorySettings()
    file: FileSettings = FileSettings()
    statistical: StatisticalSettings = StatisticalSettings()
    integrity: IntegritySettings = IntegritySettings()
    
    def __post_init__(self):
        """Initialize with default settings if not provided."""
        if not hasattr(self, 'memory') or self.memory is None:
            self.memory = MemorySettings()
        if not hasattr(self, 'file') or self.file is None:
            self.file = FileSettings()
        if not hasattr(self, 'statistical') or self.statistical is None:
            self.statistical = StatisticalSettings()
        if not hasattr(self, 'integrity') or self.integrity is None:
            self.integrity = IntegritySettings()


# Global settings instance
_settings: Optional[RLDebugKitSettings] = None


def get_settings() -> RLDebugKitSettings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = RLDebugKitSettings()
    return _settings


def set_settings(settings: RLDebugKitSettings) -> None:
    """Set the global settings instance."""
    global _settings
    _settings = settings


def reset_settings() -> None:
    """Reset settings to defaults."""
    global _settings
    _settings = RLDebugKitSettings()