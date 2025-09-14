"""Central configuration management for RLDK."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings


class RLDKSettings(BaseSettings):
    """Main RLDK configuration."""

    model_config = ConfigDict(
        env_prefix='RLDK_',
        case_sensitive=False,
        validate_assignment=True
    )

    # Output directories
    default_output_dir: Path = Field(default="rldk_reports", description="Default output directory")
    runs_dir: Path = Field(default="./runs", description="Default runs directory")
    cache_dir: Path = Field(default="./.rldk_cache", description="Cache directory")

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_file: Optional[Path] = Field(default=None, description="Log file path")
    log_to_console: bool = Field(default=True, description="Enable console logging")

    # Analysis thresholds
    default_tolerance: float = Field(default=0.01, description="Default tolerance for comparisons")
    default_window_size: int = Field(default=50, description="Default window size for analysis")
    max_episodes: Optional[int] = Field(default=None, description="Maximum episodes to process")

    # W&B settings
    wandb_project: str = Field(default="rldk-experiments", description="Default W&B project")
    wandb_enabled: bool = Field(default=True, description="Enable W&B by default")
    wandb_entity: Optional[str] = Field(default=None, description="W&B entity")
    wandb_tags: List[str] = Field(default_factory=list, description="Default W&B tags")

    # Performance settings
    num_workers: int = Field(default=4, description="Number of parallel workers")
    batch_size: int = Field(default=32, description="Default batch size")
    memory_limit_gb: Optional[float] = Field(default=None, description="Memory limit in GB")

    # Visualization settings
    plot_style: str = Field(default="seaborn-v0_8", description="Matplotlib style")
    figure_size: tuple = Field(default=(10, 6), description="Default figure size")
    dpi: int = Field(default=100, description="Figure DPI")

    # Environment settings
    seed: Optional[int] = Field(default=None, description="Random seed")
    debug: bool = Field(default=False, description="Debug mode")

    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()

    @field_validator('default_tolerance')
    @classmethod
    def validate_tolerance(cls, v):
        """Validate tolerance value."""
        if not 0 < v < 1:
            raise ValueError("tolerance must be between 0 and 1")
        return v

    @field_validator('default_window_size')
    @classmethod
    def validate_window_size(cls, v):
        """Validate window size."""
        if v <= 0:
            raise ValueError("window_size must be positive")
        return v

    @field_validator('num_workers')
    @classmethod
    def validate_num_workers(cls, v):
        """Validate number of workers."""
        if v <= 0:
            raise ValueError("num_workers must be positive")
        return v

    @field_validator('batch_size')
    @classmethod
    def validate_batch_size(cls, v):
        """Validate batch size."""
        if v <= 0:
            raise ValueError("batch_size must be positive")
        return v

    def setup_logging(self) -> None:
        """Setup logging configuration."""
        # Create log directory if log_file is specified
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Configure logging
        handlers = []

        if self.log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, self.log_level))
            console_handler.setFormatter(logging.Formatter(self.log_format))
            handlers.append(console_handler)

        if self.log_file:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(getattr(logging, self.log_level))
            file_handler.setFormatter(logging.Formatter(self.log_format))
            handlers.append(file_handler)

        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format=self.log_format,
            handlers=handlers,
            force=True
        )

    def create_directories(self) -> None:
        """Create necessary directories."""
        self.default_output_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_wandb_config(self) -> Dict[str, Any]:
        """Get W&B configuration dictionary."""
        config = {
            "project": self.wandb_project,
            "tags": self.wandb_tags,
        }
        if self.wandb_entity:
            config["entity"] = self.wandb_entity
        return config

    def initialize(self) -> None:
        """Initialize logging and create directories.

        This method should be called explicitly when the configuration
        needs to be fully initialized. It's not called automatically
        on import to avoid side effects.
        """
        self.setup_logging()
        self.create_directories()


# Global settings instance - initialized lazily
_settings = None

def get_settings() -> RLDKSettings:
    """Get global settings instance, initializing if needed."""
    global _settings
    if _settings is None:
        _settings = RLDKSettings()
    return _settings

class _SettingsProxy:
    def __getattr__(self, name):
        return getattr(get_settings(), name)

    def __setattr__(self, name, value):
        setattr(get_settings(), name, value)

    def __dir__(self):
        return dir(get_settings())

    def __repr__(self):
        return repr(get_settings())

    def __str__(self):
        return str(get_settings())

    def __eq__(self, other):
        return get_settings() == other

    def __hash__(self):
        return hash(get_settings())

    def __bool__(self):
        return bool(get_settings())

    def __getitem__(self, key):
        return getattr(get_settings(), key)

    def __setitem__(self, key, value):
        setattr(get_settings(), key, value)

    @property
    def __class__(self):
        return get_settings().__class__

settings = _SettingsProxy()

# Note: setup_logging() and create_directories() are not called automatically
# to avoid side effects on import. Call them explicitly when needed:
# settings.setup_logging()
# settings.create_directories()
