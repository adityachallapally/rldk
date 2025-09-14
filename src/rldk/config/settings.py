"""Central configuration management for RLDK."""

import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings

from ..utils.runtime import with_timeout


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
    
    # Tracking performance settings
    tracking_timeout: float = Field(default=30.0, description="Default timeout for tracking operations in seconds")
    dataset_sample_size: int = Field(default=1000, description="Sample size for large dataset checksums")
    model_fingerprint_limit: int = Field(default=100000000, description="Parameter limit for model fingerprinting (100M)")
    enable_async_init: bool = Field(default=True, description="Enable async initialization for tracking components")
    cache_environment: bool = Field(default=True, description="Cache environment information")
    cache_git_info: bool = Field(default=True, description="Cache git information")
    git_timeout: float = Field(default=10.0, description="Timeout for git operations in seconds")
    environment_timeout: float = Field(default=30.0, description="Timeout for environment capture in seconds")

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

    @field_validator('tracking_timeout')
    @classmethod
    def validate_tracking_timeout(cls, v):
        """Validate tracking timeout."""
        if v <= 0:
            raise ValueError("tracking_timeout must be positive")
        return v

    @field_validator('dataset_sample_size')
    @classmethod
    def validate_dataset_sample_size(cls, v):
        """Validate dataset sample size."""
        if v <= 0:
            raise ValueError("dataset_sample_size must be positive")
        return v

    @field_validator('model_fingerprint_limit')
    @classmethod
    def validate_model_fingerprint_limit(cls, v):
        """Validate model fingerprint limit."""
        if v <= 0:
            raise ValueError("model_fingerprint_limit must be positive")
        return v

    @field_validator('git_timeout')
    @classmethod
    def validate_git_timeout(cls, v):
        """Validate git timeout."""
        if v <= 0:
            raise ValueError("git_timeout must be positive")
        return v

    @field_validator('environment_timeout')
    @classmethod
    def validate_environment_timeout(cls, v):
        """Validate environment timeout."""
        if v <= 0:
            raise ValueError("environment_timeout must be positive")
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

    @lru_cache(maxsize=1)
    def get_cache_dir(self) -> Path:
        """Get cache directory with lazy creation."""
        cache_dir = self.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    @lru_cache(maxsize=1)
    def get_output_dir(self) -> Path:
        """Get output directory with lazy creation."""
        output_dir = self.default_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @lru_cache(maxsize=1)
    def get_runs_dir(self) -> Path:
        """Get runs directory with lazy creation."""
        runs_dir = self.runs_dir
        runs_dir.mkdir(parents=True, exist_ok=True)
        return runs_dir

    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance-related configuration."""
        return {
            "tracking_timeout": self.tracking_timeout,
            "dataset_sample_size": self.dataset_sample_size,
            "model_fingerprint_limit": self.model_fingerprint_limit,
            "enable_async_init": self.enable_async_init,
            "cache_environment": self.cache_environment,
            "cache_git_info": self.cache_git_info,
            "git_timeout": self.git_timeout,
            "environment_timeout": self.environment_timeout,
            "num_workers": self.num_workers,
            "memory_limit_gb": self.memory_limit_gb
        }

    @with_timeout(5.0)
    def _safe_create_directories(self) -> None:
        """Safely create directories with timeout."""
        self.create_directories()

    def initialize_async(self) -> None:
        """Initialize settings asynchronously with performance optimizations."""
        try:
            # Use cached directory creation
            self.get_cache_dir()
            self.get_output_dir()
            self.get_runs_dir()
            
            # Setup logging with timeout
            self.setup_logging()
        except Exception as e:
            logging.warning(f"Async initialization failed, falling back to sync: {e}")
            self.initialize()


# Global settings instance - initialized lazily
_settings = None

def get_settings() -> RLDKSettings:
    """Get global settings instance, initializing if needed."""
    global _settings
    if _settings is None:
        _settings = RLDKSettings()
        # Use async initialization if enabled
        if _settings.enable_async_init:
            _settings.initialize_async()
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
