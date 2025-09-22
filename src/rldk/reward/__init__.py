"""Reward module for RL Debug Kit."""

# Create alias for backward compatibility
from . import health_analysis as health_module
from .api import HealthAnalysisResult, reward_health
from .calibration import analyze_calibration
from .drift import compare_models, detect_reward_drift
from .length_bias import LengthBiasDetector, LengthBiasMetrics
from .health_analysis import OveroptimizationAnalysis, RewardHealthReport, health

__all__ = [
    "health",
    "reward_health",
    "RewardHealthReport",
    "OveroptimizationAnalysis",
    "HealthAnalysisResult",
    "compare_models",
    "detect_reward_drift",
    "analyze_calibration",
    "health_module",
    "LengthBiasDetector",
    "LengthBiasMetrics",
]
