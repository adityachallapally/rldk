"""Configuration for forensics and analysis parameters."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

@dataclass
class ForensicsConfig:
    """Configuration for forensics and analysis parameters."""

    # Advantage statistics tracking
    ADVANTAGE_WINDOW_SIZE: int = 100
    ADVANTAGE_TREND_WINDOW: int = 20
    ADVANTAGE_BIAS_THRESHOLD: float = 0.1
    ADVANTAGE_SCALE_THRESHOLD: float = 2.0
    ADVANTAGE_DISTRIBUTION_WINDOW: int = 50
    ADVANTAGE_SAMPLES_MULTIPLIER: int = 10

    # Gradient analysis
    GRADIENT_NORM_WINDOW: int = 50
    GRADIENT_EXPLOSION_THRESHOLD: float = 10.0
    GRADIENT_VANISHING_THRESHOLD: float = 1e-6
    GRADIENT_STABILITY_THRESHOLD: float = 1.0

    # KL divergence tracking
    KL_WINDOW_SIZE: int = 100
    KL_ANOMALY_THRESHOLD: float = 0.5
    KL_TREND_WINDOW: int = 20
    KL_STABILITY_THRESHOLD: float = 0.1

    # PPO scan parameters
    PPO_SCAN_WINDOW_SIZE: int = 50
    PPO_ANOMALY_THRESHOLD: float = 0.3
    PPO_CONSISTENCY_THRESHOLD: float = 0.8
    PPO_DRIFT_THRESHOLD: float = 0.2

    # Checkpoint diff analysis
    CKPT_DIFF_THRESHOLD: float = 0.01
    CKPT_SIGNIFICANCE_LEVEL: float = 0.05
    CKPT_MIN_SAMPLES: int = 10

    # Environment audit
    ENV_AUDIT_SAMPLE_SIZE: int = 1000
    ENV_AUDIT_CONFIDENCE_LEVEL: float = 0.95
    ENV_AUDIT_TOLERANCE: float = 0.01

    # Log scan parameters
    LOG_SCAN_WINDOW_SIZE: int = 100
    LOG_ANOMALY_THRESHOLD: float = 0.1
    LOG_ERROR_THRESHOLD: float = 0.05

    # Statistical analysis
    MIN_SAMPLES_FOR_STATS: int = 3
    MIN_SAMPLES_FOR_DISTRIBUTION: int = 20
    MIN_SAMPLES_FOR_CORRELATION: int = 10
    MIN_SAMPLES_FOR_TREND: int = 5

    # Percentile analysis
    PERCENTILES: List[int] = None

    # Anomaly detection
    ANOMALY_SEVERITY_CRITICAL: float = 0.8
    ANOMALY_SEVERITY_WARNING: float = 0.5
    ANOMALY_SEVERITY_INFO: float = 0.3

    # Health scoring weights
    HEALTH_WEIGHT_NORMALIZATION: float = 0.4
    HEALTH_WEIGHT_BIAS: float = 0.3
    HEALTH_WEIGHT_SCALE: float = 0.2
    HEALTH_WEIGHT_DISTRIBUTION: float = 0.1

    # Quality scoring weights
    QUALITY_WEIGHT_SCALE_STABILITY: float = 0.3
    QUALITY_WEIGHT_MEAN_TREND: float = 0.3
    QUALITY_WEIGHT_VOLATILITY: float = 0.2
    QUALITY_WEIGHT_SKEWNESS: float = 0.2

    # Trend analysis
    TREND_POLYFIT_DEGREE: int = 1
    TREND_SLOPE_MULTIPLIER: float = 10.0
    TREND_VOLATILITY_MULTIPLIER: float = 5.0
    TREND_SKEWNESS_DIVISOR: float = 2.0

    # Distribution analysis
    DISTRIBUTION_SKEWNESS_THRESHOLD: float = 2.0
    DISTRIBUTION_KURTOSIS_THRESHOLD: float = 4.0
    DISTRIBUTION_RISK_MULTIPLIER: float = 4.0

    # Coefficient of variation thresholds
    CV_STABILITY_THRESHOLD: float = 0.1
    CV_ANOMALY_THRESHOLD: float = 0.5

    # Correlation analysis
    CORRELATION_MIN_SAMPLES: int = 10
    CORRELATION_SIGNIFICANCE_LEVEL: float = 0.05

    # Bootstrap analysis
    BOOTSTRAP_SAMPLES: int = 1000
    BOOTSTRAP_CONFIDENCE_LEVEL: float = 0.95

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
DEFAULT_FORENSICS_CONFIG = ForensicsConfig()

# Environment-specific configurations
CONFIGS = {
    "default": DEFAULT_FORENSICS_CONFIG,
    "strict": ForensicsConfig(
        ADVANTAGE_BIAS_THRESHOLD=0.05,
        ADVANTAGE_SCALE_THRESHOLD=1.5,
        GRADIENT_EXPLOSION_THRESHOLD=5.0,
        KL_ANOMALY_THRESHOLD=0.3,
        PPO_ANOMALY_THRESHOLD=0.2,
        MIN_SAMPLES_FOR_STATS=5,
    ),
    "lenient": ForensicsConfig(
        ADVANTAGE_BIAS_THRESHOLD=0.2,
        ADVANTAGE_SCALE_THRESHOLD=3.0,
        GRADIENT_EXPLOSION_THRESHOLD=20.0,
        KL_ANOMALY_THRESHOLD=0.8,
        PPO_ANOMALY_THRESHOLD=0.5,
        MIN_SAMPLES_FOR_STATS=2,
    ),
    "research": ForensicsConfig(
        ADVANTAGE_BIAS_THRESHOLD=0.01,
        ADVANTAGE_SCALE_THRESHOLD=1.0,
        GRADIENT_EXPLOSION_THRESHOLD=2.0,
        KL_ANOMALY_THRESHOLD=0.1,
        PPO_ANOMALY_THRESHOLD=0.1,
        MIN_SAMPLES_FOR_STATS=1,
        BOOTSTRAP_SAMPLES=5000,
    ),
}

def get_forensics_config(config_name: str = "default") -> ForensicsConfig:
    """Get forensics configuration by name."""
    if config_name not in CONFIGS:
        logger.warning(f"Unknown config name '{config_name}', using default")
        return DEFAULT_FORENSICS_CONFIG
    return CONFIGS[config_name]

def create_custom_forensics_config(**kwargs) -> ForensicsConfig:
    """Create a custom forensics configuration with overridden values."""
    config = ForensicsConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown forensics configuration parameter: {key}")
    return config
