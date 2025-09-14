"""Configuration management for RLDK."""

from .settings import settings, RLDKSettings
from .schemas import ConfigSchema, AnalysisConfig, LoggingConfig, WandBConfig
from .evaluation_config import (
    EvaluationConfig,
    DEFAULT_EVAL_CONFIG,
    get_eval_config,
    load_config_from_env as load_eval_config_from_env,
    create_custom_config as create_custom_eval_config,
)
from .forensics_config import (
    ForensicsConfig,
    DEFAULT_FORENSICS_CONFIG,
    get_forensics_config,
    create_custom_forensics_config,
)
from .visualization_config import (
    VisualizationConfig,
    DEFAULT_VISUALIZATION_CONFIG,
    get_visualization_config,
    create_custom_visualization_config,
)
from .suite_config import (
    SuiteConfig,
    DEFAULT_SUITE_CONFIG,
    get_suite_config,
    create_custom_suite_config,
)
from .validator import (
    ConfigValidator,
    validate_all_configs,
    print_validation_results,
)

__all__ = [
    # Main settings
    "settings",
    "RLDKSettings",
    # Legacy schemas
    "ConfigSchema",
    "AnalysisConfig",
    "LoggingConfig",
    "WandBConfig",
    # Evaluation config
    "EvaluationConfig",
    "DEFAULT_EVAL_CONFIG",
    "get_eval_config",
    "load_eval_config_from_env",
    "create_custom_eval_config",
    # Forensics config
    "ForensicsConfig",
    "DEFAULT_FORENSICS_CONFIG",
    "get_forensics_config",
    "create_custom_forensics_config",
    # Visualization config
    "VisualizationConfig",
    "DEFAULT_VISUALIZATION_CONFIG",
    "get_visualization_config",
    "create_custom_visualization_config",
    # Suite config
    "SuiteConfig",
    "DEFAULT_SUITE_CONFIG",
    "get_suite_config",
    "create_custom_suite_config",
    # Validator
    "ConfigValidator",
    "validate_all_configs",
    "print_validation_results",
]