"""Reward module for RL Debug Kit."""

# Create alias for backward compatibility
from . import health_analysis as health_module
from .calibration import analyze_calibration
from .drift import compare_models, detect_reward_drift
from .health_analysis import RewardHealthReport, health

__all__ = ["health", "RewardHealthReport", "compare_models", "detect_reward_drift", "analyze_calibration", "health_module"]
