"""Extended configuration settings for RL Debug Kit bug fixes."""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from .schemas import ConfigSchema


class MemorySettings(BaseModel):
    """Memory management settings with validation."""
    max_parameter_size_mb: int = Field(default=50, gt=0, le=1000, description="Maximum parameter size in MB")
    max_total_memory_mb: int = Field(default=500, gt=0, le=10000, description="Maximum total memory usage in MB")
    cleanup_threshold_mb: int = Field(default=100, gt=0, le=1000, description="Memory threshold for cleanup in MB")
    
    @field_validator('max_total_memory_mb')
    @classmethod
    def validate_total_memory(cls, v, info):
        """Ensure total memory is greater than parameter size."""
        if 'max_parameter_size_mb' in info.data and v <= info.data['max_parameter_size_mb']:
            raise ValueError("max_total_memory_mb must be greater than max_parameter_size_mb")
        return v


class FileSettings(BaseModel):
    """File operation settings with validation."""
    max_file_size_mb: int = Field(default=5, gt=0, le=100, description="Maximum file size to process in MB")
    max_read_size_kb: int = Field(default=512, gt=0, le=10240, description="Maximum bytes to read from file in KB")
    max_files_to_process: int = Field(default=50, gt=0, le=1000, description="Maximum number of files to process")
    encoding: str = Field(default="utf-8", description="File encoding for text files")
    
    @field_validator('encoding')
    @classmethod
    def validate_encoding(cls, v):
        """Validate encoding is supported."""
        try:
            "test".encode(v)
            return v
        except (LookupError, UnicodeError):
            raise ValueError(f"Unsupported encoding: {v}")


class StatisticalSettings(BaseModel):
    """Statistical calculation settings with validation."""
    default_confidence_level: float = Field(default=0.95, gt=0, lt=1, description="Default confidence level")
    conservative_std_estimate: float = Field(default=0.3, gt=0, le=1, description="Conservative standard deviation estimate")
    binomial_threshold: float = Field(default=0.1, gt=0, le=1, description="Threshold for binomial approximation")
    cv_threshold: float = Field(default=0.5, gt=0, le=2, description="Coefficient of variation threshold")
    
    @field_validator('default_confidence_level')
    @classmethod
    def validate_confidence_level(cls, v):
        """Validate confidence level is reasonable."""
        if v < 0.5 or v > 0.99:
            raise ValueError("Confidence level should be between 0.5 and 0.99")
        return v


class IntegritySettings(BaseModel):
    """Integrity check settings with validation."""
    duplicate_threshold: float = Field(default=0.1, gt=0, le=1, description="Threshold for duplicate detection")
    pattern_threshold: float = Field(default=0.3, gt=0, le=1, description="Threshold for pattern detection")
    leakage_threshold: float = Field(default=0.05, gt=0, le=1, description="Threshold for data leakage")
    max_penalty_per_check: float = Field(default=0.3, gt=0, le=1, description="Maximum penalty per integrity check")


class BugFixSettings(BaseModel):
    """Settings for bug fixes and new features."""
    enable_data_driven_stats: bool = Field(default=True, description="Enable data-driven statistical methods")
    enable_memory_management: bool = Field(default=True, description="Enable improved memory management")
    enable_file_safety: bool = Field(default=True, description="Enable enhanced file safety measures")
    enable_legacy_methods: bool = Field(default=True, description="Enable legacy method fallbacks")
    
    # Feature flags for gradual rollout
    use_new_confidence_intervals: bool = Field(default=False, description="Use new confidence interval methods")
    use_new_effect_sizes: bool = Field(default=False, description="Use new effect size calculations")
    use_new_calibration: bool = Field(default=False, description="Use new calibration scoring")


class ExtendedConfigSchema(ConfigSchema):
    """Extended configuration schema that includes bug fix settings."""
    
    # New settings for bug fixes
    memory: MemorySettings = Field(default_factory=MemorySettings)
    file: FileSettings = Field(default_factory=FileSettings)
    statistical: StatisticalSettings = Field(default_factory=StatisticalSettings)
    integrity: IntegritySettings = Field(default_factory=IntegritySettings)
    bug_fixes: BugFixSettings = Field(default_factory=BugFixSettings)
    
    def get_bug_fix_summary(self) -> dict:
        """Get summary of bug fix settings."""
        return {
            "data_driven_stats": self.bug_fixes.enable_data_driven_stats,
            "memory_management": self.bug_fixes.enable_memory_management,
            "file_safety": self.bug_fixes.enable_file_safety,
            "legacy_methods": self.bug_fixes.enable_legacy_methods,
            "new_confidence_intervals": self.bug_fixes.use_new_confidence_intervals,
            "new_effect_sizes": self.bug_fixes.use_new_effect_sizes,
            "new_calibration": self.bug_fixes.use_new_calibration,
        }


# Global settings instance
_settings: Optional[ExtendedConfigSchema] = None


def get_settings() -> ExtendedConfigSchema:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = ExtendedConfigSchema()
    return _settings


def set_settings(settings: ExtendedConfigSchema) -> None:
    """Set the global settings instance."""
    global _settings
    _settings = settings


def reset_settings() -> None:
    """Reset settings to defaults."""
    global _settings
    _settings = ExtendedConfigSchema()


def load_settings_from_env() -> ExtendedConfigSchema:
    """Load settings from environment variables."""
    import os
    
    settings = ExtendedConfigSchema()
    
    # Override with environment variables if present
    if os.getenv("RLDK_MAX_PARAMETER_SIZE_MB"):
        settings.memory.max_parameter_size_mb = int(os.getenv("RLDK_MAX_PARAMETER_SIZE_MB"))
    
    if os.getenv("RLDK_MAX_FILE_SIZE_MB"):
        settings.file.max_file_size_mb = int(os.getenv("RLDK_MAX_FILE_SIZE_MB"))
    
    if os.getenv("RLDK_USE_NEW_STATS"):
        settings.bug_fixes.use_new_confidence_intervals = os.getenv("RLDK_USE_NEW_STATS").lower() == "true"
        settings.bug_fixes.use_new_effect_sizes = os.getenv("RLDK_USE_NEW_STATS").lower() == "true"
    
    return settings