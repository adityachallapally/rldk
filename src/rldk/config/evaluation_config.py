"""Centralized configuration for evaluation parameters."""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    """Configuration for evaluation parameters."""

    # KL Divergence thresholds
    KL_DIVERGENCE_MIN: float = 0.01
    KL_DIVERGENCE_MAX: float = 0.5
    KL_DIVERGENCE_TARGET: float = 0.1

    # Improvement score normalization
    IMPROVEMENT_RANGE_MIN: float = -1.0
    IMPROVEMENT_RANGE_MAX: float = 1.0

    # Loss score normalization
    MAX_LOSS_THRESHOLD: float = 10.0

    # Memory thresholds (GB)
    MEMORY_EFFICIENCY_THRESHOLD: float = 8.0
    MEMORY_STABILITY_THRESHOLD: float = 1.0
    GPU_MEMORY_EFFICIENCY_THRESHOLD: float = 6.0
    MEMORY_RANGE_THRESHOLD: float = 2.0
    MEMORY_CONSISTENCY_THRESHOLD: float = 16.0

    # Gradient thresholds
    GRADIENT_STABILITY_THRESHOLD: float = 1.0
    GRADIENT_EXPLOSION_THRESHOLD: float = 10.0
    GRADIENT_EFFICIENCY_THRESHOLD: float = 4.0

    # Toxicity thresholds
    HIGH_TOXICITY_THRESHOLD: float = 0.7
    CONFIDENCE_CALIBRATION_MIN: float = 0.3
    CONFIDENCE_CALIBRATION_MAX: float = 0.8
    CONFIDENCE_STABILITY_THRESHOLD: float = 0.2

    # Performance thresholds
    INFERENCE_TIME_THRESHOLD: float = 0.1  # seconds
    LATENCY_THRESHOLD: float = 0.05  # seconds
    STEPS_PER_SECOND_MAX: float = 1000.0
    SPEED_CONSISTENCY_THRESHOLD: float = 0.01
    BATCH_SPEED_THRESHOLD: float = 1000.0

    # Consistency thresholds
    CV_CONSISTENCY_THRESHOLD: float = 0.2
    OUTLIER_THRESHOLD_MULTIPLIER: float = 1.5
    STABILITY_THRESHOLD: float = 0.1

    # Robustness thresholds
    TREND_DEGRADATION_THRESHOLD: float = 0.1
    MAX_EXPECTED_DEGRADATION: float = 0.1
    LOW_ROBUSTNESS_THRESHOLD: float = 0.3

    # Efficiency thresholds
    CONVERGENCE_IMPROVEMENT_THRESHOLD: float = 0.5
    EARLY_CONVERGENCE_THRESHOLD: float = 0.9
    FLOP_EFFICIENCY_THRESHOLD: float = 1000.0
    SAMPLE_EFFICIENCY_MAX: float = 2.0

    # Calibration thresholds
    UNCERTAINTY_CALIBRATION_MIN: float = 0.1
    UNCERTAINTY_CALIBRATION_MAX: float = 0.5
    ENTROPY_CALIBRATION_MIN: float = 0.5
    ENTROPY_CALIBRATION_MAX: float = 2.0
    TEMPERATURE_CALIBRATION_MIN: float = 0.8
    TEMPERATURE_CALIBRATION_MAX: float = 1.2
    ECE_THRESHOLD: float = 0.1

    # Memory access and allocation
    MEMORY_ACCESS_THRESHOLD: float = 0.001  # seconds
    ALLOCATION_EFFICIENCY_THRESHOLD: float = 1000
    BANDWIDTH_EFFICIENCY_MIN: float = 0.3
    BANDWIDTH_EFFICIENCY_MAX: float = 0.8

    # GPU utilization
    GPU_UTILIZATION_MIN: float = 0.3
    GPU_UTILIZATION_MAX: float = 0.8

    # Sample size thresholds
    MIN_SAMPLES_FOR_ANALYSIS: int = 10
    MIN_SAMPLES_FOR_CONSISTENCY: int = 5
    MIN_SAMPLES_FOR_DISTRIBUTION: int = 50
    MIN_SAMPLES_FOR_TREND: int = 20

    # Prompt analysis thresholds
    PROMPT_LENGTH_SHORT: int = 50
    PROMPT_LENGTH_MEDIUM: int = 150
    PROMPT_START_CHARS: int = 20

    # Percentile thresholds
    PERCENTILES: List[int] = None

    # Correlation thresholds
    MIN_SAMPLES_FOR_CORRELATION: int = 10

    # Bootstrap confidence level
    BOOTSTRAP_CONFIDENCE_LEVEL: float = 0.95

    # Catastrophic forgetting thresholds
    CATASTROPHIC_REGRESSION_THRESHOLD: float = -0.05
    CATASTROPHIC_REGRESSION_Z_THRESHOLD: float = -2.0
    CATASTROPHIC_MIN_SAMPLES: int = 5
    CATASTROPHIC_WEIGHTING_STRATEGY: str = "baseline_count"

    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.PERCENTILES is None:
            self.PERCENTILES = [5, 10, 25, 50, 75, 90, 95]

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

# Default configuration instance
DEFAULT_EVAL_CONFIG = EvaluationConfig()

# Environment-specific configurations
CONFIGS = {
    "default": DEFAULT_EVAL_CONFIG,
    "strict": EvaluationConfig(
        KL_DIVERGENCE_MAX=0.3,
        HIGH_TOXICITY_THRESHOLD=0.5,
        MEMORY_EFFICIENCY_THRESHOLD=6.0,
        CONFIDENCE_CALIBRATION_MIN=0.4,
        CONFIDENCE_CALIBRATION_MAX=0.7,
        MIN_SAMPLES_FOR_ANALYSIS=20,
    ),
    "lenient": EvaluationConfig(
        KL_DIVERGENCE_MAX=0.8,
        HIGH_TOXICITY_THRESHOLD=0.9,
        MEMORY_EFFICIENCY_THRESHOLD=12.0,
        CONFIDENCE_CALIBRATION_MIN=0.2,
        CONFIDENCE_CALIBRATION_MAX=0.9,
        MIN_SAMPLES_FOR_ANALYSIS=5,
    ),
    "research": EvaluationConfig(
        KL_DIVERGENCE_MAX=1.0,
        HIGH_TOXICITY_THRESHOLD=0.8,
        MEMORY_EFFICIENCY_THRESHOLD=16.0,
        CONFIDENCE_CALIBRATION_MIN=0.1,
        CONFIDENCE_CALIBRATION_MAX=0.9,
        MIN_SAMPLES_FOR_ANALYSIS=3,
        MIN_SAMPLES_FOR_CONSISTENCY=2,
    ),
}

def get_eval_config(config_name: str = "default") -> EvaluationConfig:
    """Get evaluation configuration by name."""
    if config_name not in CONFIGS:
        logger.warning(f"Unknown config name '{config_name}', using default")
        return DEFAULT_EVAL_CONFIG
    return CONFIGS[config_name]

def load_config_from_env() -> EvaluationConfig:
    """Load configuration from environment variables."""
    config = EvaluationConfig()

    # Override with environment variables if they exist
    for field in config.__dataclass_fields__:
        env_var = f"RLDK_{field}"
        if env_var in os.environ:
            value = os.environ[env_var]
            # Try to convert to appropriate type
            field_type = config.__dataclass_fields__[field].type
            try:
                if field_type == float:
                    setattr(config, field, float(value))
                elif field_type == int:
                    setattr(config, field, int(value))
                elif field_type == str:
                    setattr(config, field, str(value))
                elif field_type == bool:
                    setattr(config, field, value.lower() in ('true', '1', 'yes', 'on'))
                elif field_type == List[int]:
                    # Handle list of integers
                    if value.startswith('[') and value.endswith(']'):
                        # Parse as JSON-like list
                        import json
                        setattr(config, field, json.loads(value))
                    else:
                        # Parse as comma-separated values
                        setattr(config, field, [int(x.strip()) for x in value.split(',')])
                else:
                    logger.warning(f"Unsupported field type {field_type} for {field}")
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse environment variable {env_var}={value}: {e}")

    return config

def create_custom_config(**kwargs) -> EvaluationConfig:
    """Create a custom configuration with overridden values."""
    config = EvaluationConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown configuration parameter: {key}")
    return config
