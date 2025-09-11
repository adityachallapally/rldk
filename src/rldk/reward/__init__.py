"""Reward module for RL Debug Kit."""

from .health_analysis import health, RewardHealthReport
from .drift import compare_models, detect_reward_drift
from .calibration import analyze_calibration

# Create alias for backward compatibility
from . import health_analysis as health_module

__all__ = ["health", "RewardHealthReport", "compare_models", "detect_reward_drift", "analyze_calibration", "health_module"]
