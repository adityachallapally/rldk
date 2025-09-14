"""Configuration management for RLDK."""

from .evaluation_config import (
    DEFAULT_EVAL_CONFIG,
    EvaluationConfig,
    get_eval_config,
)
from .evaluation_config import (
    create_custom_config as create_custom_eval_config,
)
from .evaluation_config import (
    load_config_from_env as load_eval_config_from_env,
)
from .forensics_config import (
    DEFAULT_FORENSICS_CONFIG,
    ForensicsConfig,
    create_custom_forensics_config,
    get_forensics_config,
)
from .schemas import AnalysisConfig, ConfigSchema, LoggingConfig, WandBConfig
from .settings import RLDKSettings, settings
from .suite_config import (
    DEFAULT_SUITE_CONFIG,
    SuiteConfig,
    create_custom_suite_config,
    get_suite_config,
)
from .validator import (
    ConfigValidator,
    print_validation_results,
    validate_all_configs,
)
from .visualization_config import (
    DEFAULT_VISUALIZATION_CONFIG,
    VisualizationConfig,
    create_custom_visualization_config,
    get_visualization_config,
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
