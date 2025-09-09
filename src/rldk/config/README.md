# RLDK Configuration System

This directory contains the centralized configuration management system for RLDK.

## Overview

The configuration system provides:
- Centralized settings management using Pydantic
- Environment variable support with `RLDK_` prefix
- Type validation and automatic conversion
- Hierarchical configuration schemas
- Default values for all settings

## Files

- `settings.py` - Main settings class and global instance
- `schemas.py` - Configuration schemas for different components
- `__init__.py` - Module exports

## Usage

### Basic Usage

```python
from rldk.config import settings

# Access global settings
print(f"Log level: {settings.log_level}")
print(f"W&B project: {settings.wandb_project}")
print(f"Default tolerance: {settings.default_tolerance}")
```

### Custom Settings

```python
from rldk.config import RLDKSettings

# Create custom settings instance
custom_settings = RLDKSettings(
    log_level="DEBUG",
    wandb_enabled=False,
    default_tolerance=0.05,
    wandb_project="my-project"
)
```

### Configuration Schemas

```python
from rldk.config import ConfigSchema, LoggingConfig, AnalysisConfig, WandBConfig

# Create individual configs
logging_config = LoggingConfig(level="WARNING", console=False)
analysis_config = AnalysisConfig(tolerance=0.02, window_size=100)
wandb_config = WandBConfig(project="test-project", enabled=True)

# Combine into main config
main_config = ConfigSchema(
    logging=logging_config,
    analysis=analysis_config,
    wandb=wandb_config
)

# Validate and export
print(f"Valid: {main_config.validate_config()}")
print(f"Summary: {main_config.get_summary()}")
```

### Environment Variables

You can override any setting using environment variables with the `RLDK_` prefix:

```bash
export RLDK_LOG_LEVEL=DEBUG
export RLDK_WANDB_ENABLED=false
export RLDK_DEFAULT_TOLERANCE=0.05
export RLDK_WANDB_PROJECT=my-experiment
```

## Configuration Options

### Main Settings (RLDKSettings)

- **Output directories**: `default_output_dir`, `runs_dir`, `cache_dir`
- **Logging**: `log_level`, `log_format`, `log_file`, `log_to_console`
- **Analysis**: `default_tolerance`, `default_window_size`, `max_episodes`
- **W&B**: `wandb_project`, `wandb_enabled`, `wandb_entity`, `wandb_tags`
- **Performance**: `num_workers`, `batch_size`, `memory_limit_gb`
- **Visualization**: `plot_style`, `figure_size`, `dpi`
- **Environment**: `seed`, `debug`

### Schema Components

- **LoggingConfig**: Logging-specific settings
- **AnalysisConfig**: Analysis parameters and thresholds
- **WandBConfig**: Weights & Biases configuration
- **PerformanceConfig**: Performance and resource settings
- **VisualizationConfig**: Plot and visualization settings
- **DirectoryConfig**: Directory paths and structure
- **EnvironmentConfig**: Environment and debugging settings

## Integration

The configuration system is integrated throughout RLDK:

- **CLI**: Uses settings for default values and parameter validation
- **Tracking**: Automatically uses W&B and directory settings
- **Adapters**: Use configuration for connection settings
- **Analysis**: Use tolerance and window size settings
- **Logging**: Configured when explicitly initialized

## Initialization

The configuration system defers side effects (logging setup and directory creation) to avoid breaking consumers in read-only environments. To initialize:

```python
from rldk.config import settings

# Initialize logging and create directories
settings.initialize()
```

**Important**: The CLI automatically calls `settings.initialize()` when needed, but library users should call it explicitly if they need logging or directory creation.

## Examples

See `examples/config_usage.py` for a comprehensive example of using the configuration system.