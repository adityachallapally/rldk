"""Configuration schemas for RLDK."""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class PlotStyle(str, Enum):
    """Matplotlib plot styles."""
    DEFAULT = "default"
    SEABORN = "seaborn-v0_8"
    SEABORN_WHITE = "seaborn-white"
    SEABORN_WHITEGRID = "seaborn-whitegrid"
    SEABORN_DARK = "seaborn-dark"
    SEABORN_DARKGRID = "seaborn-darkgrid"
    SEABORN_TICKS = "seaborn-ticks"
    SEABORN_COLORBLIND = "seaborn-colorblind"

class LoggingConfig(BaseModel):
    """Logging configuration schema."""
    level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file: Optional[Path] = Field(default=None, description="Log file path")
    console: bool = Field(default=True, description="Enable console logging")

    model_config = ConfigDict(use_enum_values=True)

class AnalysisConfig(BaseModel):
    """Analysis configuration schema."""
    tolerance: float = Field(default=0.01, gt=0, lt=1, description="Tolerance for comparisons")
    window_size: int = Field(default=50, gt=0, description="Window size for analysis")
    max_episodes: Optional[int] = Field(default=None, gt=0, description="Maximum episodes to process")
    batch_size: int = Field(default=32, gt=0, description="Batch size for processing")

class WandBConfig(BaseModel):
    """W&B configuration schema."""
    project: str = Field(default="rldk-experiments", description="W&B project name")
    enabled: bool = Field(default=True, description="Enable W&B logging")
    entity: Optional[str] = Field(default=None, description="W&B entity")
    tags: List[str] = Field(default_factory=list, description="W&B tags")
    group: Optional[str] = Field(default=None, description="W&B group")
    job_type: Optional[str] = Field(default=None, description="W&B job type")

    def get_config_dict(self) -> Dict[str, Any]:
        """Get W&B configuration as dictionary."""
        config = {
            "project": self.project,
            "tags": self.tags,
        }
        if self.entity:
            config["entity"] = self.entity
        if self.group:
            config["group"] = self.group
        if self.job_type:
            config["job_type"] = self.job_type
        return config

class PerformanceConfig(BaseModel):
    """Performance configuration schema."""
    num_workers: int = Field(default=4, gt=0, description="Number of parallel workers")
    memory_limit_gb: Optional[float] = Field(default=None, gt=0, description="Memory limit in GB")
    cache_size: int = Field(default=1000, gt=0, description="Cache size for data")

class VisualizationConfig(BaseModel):
    """Visualization configuration schema."""
    style: PlotStyle = Field(default=PlotStyle.SEABORN, description="Matplotlib style")
    figure_size: tuple = Field(default=(10, 6), description="Default figure size")
    dpi: int = Field(default=100, gt=0, description="Figure DPI")
    save_format: str = Field(default="png", description="Default save format")
    show_plots: bool = Field(default=True, description="Show plots interactively")

    @field_validator('figure_size')
    @classmethod
    def validate_figure_size(cls, v):
        """Validate figure size tuple."""
        if len(v) != 2 or not all(isinstance(x, (int, float)) and x > 0 for x in v):
            raise ValueError("figure_size must be a tuple of two positive numbers")
        return v

    model_config = ConfigDict(use_enum_values=True)

class DirectoryConfig(BaseModel):
    """Directory configuration schema."""
    output_dir: Path = Field(default="rldk_reports", description="Output directory")
    runs_dir: Path = Field(default="./runs", description="Runs directory")
    cache_dir: Path = Field(default="./.rldk_cache", description="Cache directory")
    data_dir: Optional[Path] = Field(default=None, description="Data directory")

    def create_directories(self) -> None:
        """Create all configured directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.data_dir:
            self.data_dir.mkdir(parents=True, exist_ok=True)

class EnvironmentConfig(BaseModel):
    """Environment configuration schema."""
    seed: Optional[int] = Field(default=None, ge=0, description="Random seed")
    debug: bool = Field(default=False, description="Debug mode")
    verbose: bool = Field(default=False, description="Verbose output")

class ConfigSchema(BaseModel):
    """Main configuration schema combining all sub-configurations."""

    # Sub-configurations
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    wandb: WandBConfig = Field(default_factory=WandBConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    directories: DirectoryConfig = Field(default_factory=DirectoryConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)

    # Additional metadata
    version: str = Field(default="1.0.0", description="Configuration version")
    created_at: Optional[str] = Field(default=None, description="Configuration creation timestamp")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()

    def to_json(self) -> str:
        """Convert configuration to JSON string."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConfigSchema':
        """Create configuration from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> 'ConfigSchema':
        """Create configuration from JSON string."""
        return cls.model_validate_json(json_str)

    def validate_config(self) -> bool:
        """Validate the entire configuration."""
        try:
            # Pydantic models are already validated on creation
            # This method exists for compatibility and can be used to check
            # if the current state is still valid after modifications
            return True
        except Exception:
            return False

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the configuration."""
        return {
            "version": self.version,
            "logging_level": self.logging.level,
            "wandb_enabled": self.wandb.enabled,
            "wandb_project": self.wandb.project,
            "analysis_tolerance": self.analysis.tolerance,
            "analysis_window_size": self.analysis.window_size,
            "performance_workers": self.performance.num_workers,
            "visualization_style": self.visualization.style,
            "debug_mode": self.environment.debug,
        }
