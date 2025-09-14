"""Comprehensive KL schedule tracking and analysis for PPO training."""

import math
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np


def _safe_kl_value(value: Any, default: float = 0.0, max_value: float = 1e6) -> float:
    """
    Safely process KL divergence values with comprehensive edge case handling.

    Args:
        value: Input value to process
        default: Default value to use for invalid inputs
        max_value: Maximum allowed value to prevent overflow

    Returns:
        Safe float value for KL divergence
    """
    # Handle None values
    if value is None:
        return default

    # Handle string values
    if isinstance(value, str):
        try:
            value = float(value)
        except (ValueError, TypeError):
            warnings.warn(f"Could not convert string '{value}' to float, using default {default}", stacklevel=2)
            return default

    # Handle non-numeric types
    if not isinstance(value, (int, float)):
        warnings.warn(f"Non-numeric KL value type {type(value)}: {value}, using default {default}", stacklevel=2)
        return default

    # Handle NaN values
    if math.isnan(value):
        warnings.warn(f"NaN KL value detected, using default {default}", stacklevel=2)
        return default

    # Handle infinite values
    if math.isinf(value):
        if value > 0:
            warnings.warn(f"Positive infinity KL value detected, capping to {max_value}", stacklevel=2)
            return max_value
        else:
            warnings.warn(f"Negative infinity KL value detected, using default {default}", stacklevel=2)
            return default

    # Convert to float and validate range
    try:
        float_value = float(value)

        # KL divergence should be non-negative
        if float_value < 0:
            warnings.warn(f"Negative KL value {float_value} detected, using default {default}", stacklevel=2)
            return default

        # Cap extremely large values
        if float_value > max_value:
            warnings.warn(f"Extremely large KL value {float_value} detected, capping to {max_value}", stacklevel=2)
            return max_value

        return float_value

    except (ValueError, OverflowError, TypeError):
        warnings.warn(f"Error processing KL value {value}, using default {default}", stacklevel=2)
        return default


def _safe_coefficient_value(value: Any, default: float = 1.0, min_value: float = 1e-8, max_value: float = 1e6) -> float:
    """
    Safely process KL coefficient values with comprehensive edge case handling.

    Args:
        value: Input value to process
        default: Default value to use for invalid inputs
        min_value: Minimum allowed value (must be positive)
        max_value: Maximum allowed value to prevent overflow

    Returns:
        Safe float value for KL coefficient
    """
    # Handle None values
    if value is None:
        return default

    # Handle string values
    if isinstance(value, str):
        try:
            value = float(value)
        except (ValueError, TypeError):
            warnings.warn(f"Could not convert string '{value}' to float, using default {default}", stacklevel=2)
            return default

    # Handle non-numeric types
    if not isinstance(value, (int, float)):
        warnings.warn(f"Non-numeric coefficient value type {type(value)}: {value}, using default {default}", stacklevel=2)
        return default

    # Handle NaN values
    if math.isnan(value):
        warnings.warn(f"NaN coefficient value detected, using default {default}", stacklevel=2)
        return default

    # Handle infinite values
    if math.isinf(value):
        if value > 0:
            warnings.warn(f"Positive infinity coefficient value detected, capping to {max_value}", stacklevel=2)
            return max_value
        else:
            warnings.warn(f"Negative infinity coefficient value detected, using default {default}", stacklevel=2)
            return default

    # Convert to float and validate range
    try:
        float_value = float(value)

        # Coefficient should be positive
        if float_value <= 0:
            warnings.warn(f"Non-positive coefficient value {float_value} detected, using default {default}", stacklevel=2)
            return default

        # Cap extremely large values
        if float_value > max_value:
            warnings.warn(f"Extremely large coefficient value {float_value} detected, capping to {max_value}", stacklevel=2)
            return max_value

        # Ensure minimum value
        if float_value < min_value:
            warnings.warn(f"Very small coefficient value {float_value} detected, setting to minimum {min_value}", stacklevel=2)
            return min_value

        return float_value

    except (ValueError, OverflowError, TypeError):
        warnings.warn(f"Error processing coefficient value {value}, using default {default}", stacklevel=2)
        return default


@dataclass
class KLScheduleMetrics:
    """Container for KL schedule tracking metrics."""

    # Current KL metrics
    current_kl: float = 0.0
    current_kl_coef: float = 1.0
    kl_target: float = 0.1  # Default target

    # KL schedule analysis
    kl_trend: float = 0.0
    kl_volatility: float = 0.0
    kl_controller_performance: float = 1.0

    # Target range analysis
    time_in_target_range: float = 0.0
    target_range_violations: int = 0
    target_range_stability: float = 1.0

    # Controller analysis
    controller_responsiveness: float = 1.0
    controller_overshoot: float = 0.0
    controller_oscillation: float = 0.0

    # Adaptive coefficient tracking
    coef_change_rate: float = 0.0
    coef_adaptation_quality: float = 1.0
    coef_stability: float = 1.0

    # Health indicators
    kl_health_score: float = 1.0
    schedule_health_score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }


class KLScheduleTracker:
    """Advanced KL schedule tracking and analysis for PPO training."""

    def __init__(
        self,
        kl_target: float = 0.1,
        kl_target_tolerance: float = 0.05,
        window_size: int = 50,
        trend_window: int = 20,
        controller_window: int = 30,
    ):
        """Initialize KL schedule tracker.

        Args:
            kl_target: Target KL divergence value
            kl_target_tolerance: Tolerance around target for "in range" calculation
            window_size: Size of rolling window for analysis
            trend_window: Window size for trend analysis
            controller_window: Window size for controller performance analysis
        """
        self.kl_target = kl_target
        self.kl_target_tolerance = kl_target_tolerance
        self.window_size = window_size
        self.trend_window = trend_window
        self.controller_window = controller_window

        # Data storage
        self.kl_history: deque = deque(maxlen=window_size)
        self.kl_coef_history: deque = deque(maxlen=window_size)
        self.step_history: deque = deque(maxlen=window_size)

        # Analysis state
        self.current_metrics = KLScheduleMetrics(kl_target=kl_target)
        self.metrics_history: List[KLScheduleMetrics] = []

        # Controller analysis
        self.target_range_low = kl_target - kl_target_tolerance
        self.target_range_high = kl_target + kl_target_tolerance

        print(f"🎯 KL Schedule Tracker initialized - Target: {kl_target}±{kl_target_tolerance}")

    def update(self, step: int, kl_value: Union[float, Any], kl_coef: Union[float, Any]) -> KLScheduleMetrics:
        """Update tracker with new KL and coefficient values."""
        # Safely process inputs using robust edge case handling
        safe_kl_value = _safe_kl_value(kl_value, default=0.0, max_value=1e6)
        safe_kl_coef = _safe_coefficient_value(kl_coef, default=1.0, min_value=1e-8, max_value=1e6)

        # Store data
        self.kl_history.append(safe_kl_value)
        self.kl_coef_history.append(safe_kl_coef)
        self.step_history.append(step)

        # Update current metrics
        self.current_metrics.current_kl = safe_kl_value
        self.current_metrics.current_kl_coef = safe_kl_coef

        # Perform analysis if we have enough data
        if len(self.kl_history) >= 2:
            self._analyze_kl_trend()
            self._analyze_kl_volatility()
            self._analyze_target_range_performance()
            self._analyze_controller_performance()
            self._analyze_coefficient_adaptation()
            self._calculate_health_scores()

        # Store metrics
        metrics_copy = KLScheduleMetrics(**self.current_metrics.to_dict())
        self.metrics_history.append(metrics_copy)

        return metrics_copy

    def _analyze_kl_trend(self):
        """Analyze KL divergence trend over recent steps."""
        if len(self.kl_history) < self.trend_window:
            return

        recent_kl = list(self.kl_history)[-self.trend_window:]
        steps = list(range(len(recent_kl)))

        # Calculate linear trend with numerical stability checks
        if len(recent_kl) > 1:
            try:
                # Check for constant values (no trend)
                if np.std(recent_kl) < 1e-10:
                    self.current_metrics.kl_trend = 0.0
                    return

                # Use robust polynomial fitting
                trend_slope = np.polyfit(steps, recent_kl, 1)[0]

                # Validate the result
                if np.isnan(trend_slope) or np.isinf(trend_slope):
                    self.current_metrics.kl_trend = 0.0
                else:
                    # Cap extreme trend values
                    self.current_metrics.kl_trend = max(-1.0, min(1.0, float(trend_slope)))
            except (np.linalg.LinAlgError, ValueError):
                # Fallback to simple difference if polyfit fails
                if len(recent_kl) >= 2:
                    simple_trend = (recent_kl[-1] - recent_kl[0]) / len(recent_kl)
                    self.current_metrics.kl_trend = max(-1.0, min(1.0, float(simple_trend)))
                else:
                    self.current_metrics.kl_trend = 0.0

    def _analyze_kl_volatility(self):
        """Analyze KL divergence volatility."""
        if len(self.kl_history) < 10:
            return

        recent_kl = list(self.kl_history)[-10:]

        try:
            # Calculate standard deviation with numerical stability
            volatility = np.std(recent_kl, ddof=1)  # Use sample std dev

            # Validate result
            if np.isnan(volatility) or np.isinf(volatility):
                self.current_metrics.kl_volatility = 0.0
            else:
                # Cap extreme volatility values
                self.current_metrics.kl_volatility = max(0.0, min(1.0, float(volatility)))
        except (ValueError, RuntimeError):
            self.current_metrics.kl_volatility = 0.0

    def _analyze_target_range_performance(self):
        """Analyze performance within target KL range."""
        if len(self.kl_history) < 5:
            return

        recent_kl = list(self.kl_history)

        # Calculate time in target range
        in_range_count = sum(1 for kl in recent_kl
                           if self.target_range_low <= kl <= self.target_range_high)
        self.current_metrics.time_in_target_range = in_range_count / len(recent_kl)

        # Count violations
        violations = sum(1 for kl in recent_kl
                        if kl < self.target_range_low or kl > self.target_range_high)
        self.current_metrics.target_range_violations = violations

        # Calculate stability (inverse of violations)
        self.current_metrics.target_range_stability = max(0, 1 - violations / len(recent_kl))

    def _analyze_controller_performance(self):
        """Analyze KL controller performance."""
        if len(self.kl_history) < self.controller_window:
            return

        recent_kl = list(self.kl_history)[-self.controller_window:]
        recent_coef = list(self.kl_coef_history)[-self.controller_window:]

        # Controller responsiveness: how quickly coefficient changes in response to KL
        if len(recent_kl) > 5:
            try:
                kl_changes = np.diff(recent_kl)
                coef_changes = np.diff(recent_coef)

                # Calculate correlation between KL changes and coefficient changes
                if (len(kl_changes) > 1 and
                    np.std(kl_changes) > 1e-10 and
                    np.std(coef_changes) > 1e-10):

                    correlation = np.corrcoef(kl_changes, coef_changes)[0, 1]

                    # Validate correlation result
                    if np.isnan(correlation) or np.isinf(correlation):
                        self.current_metrics.controller_responsiveness = 0.0
                    else:
                        # Good responsiveness: negative correlation (coef increases when KL is high)
                        responsiveness = max(0, -correlation)
                        self.current_metrics.controller_responsiveness = min(1.0, responsiveness)
                else:
                    self.current_metrics.controller_responsiveness = 0.0
            except (ValueError, RuntimeError):
                self.current_metrics.controller_responsiveness = 0.0

        # Controller overshoot: tendency to overshoot target
        overshoots = 0
        for i in range(1, len(recent_kl)):
            prev_kl = recent_kl[i-1]
            curr_kl = recent_kl[i]

            # Check for overshoot (crossing target in wrong direction)
            if (prev_kl < self.kl_target < curr_kl) or (prev_kl > self.kl_target > curr_kl):
                overshoots += 1

        self.current_metrics.controller_overshoot = overshoots / max(1, len(recent_kl) - 1)

        # Controller oscillation: rapid back-and-forth changes
        if len(recent_coef) > 3:
            coef_changes = np.diff(recent_coef)
            sign_changes = sum(1 for i in range(1, len(coef_changes))
                             if (coef_changes[i-1] > 0) != (coef_changes[i] > 0))
            self.current_metrics.controller_oscillation = sign_changes / max(1, len(coef_changes) - 1)

    def _analyze_coefficient_adaptation(self):
        """Analyze coefficient adaptation quality."""
        if len(self.kl_coef_history) < 10:
            return

        recent_coef = list(self.kl_coef_history)[-10:]
        recent_kl = list(self.kl_history)[-10:]

        # Coefficient change rate
        coef_changes = np.diff(recent_coef)
        self.current_metrics.coef_change_rate = float(np.mean(np.abs(coef_changes)))

        # Adaptation quality: how well coefficient changes correlate with KL deviations
        try:
            kl_deviations = [abs(kl - self.kl_target) for kl in recent_kl[:-1]]
            if (len(kl_deviations) > 1 and
                np.std(kl_deviations) > 1e-10 and
                np.std(coef_changes) > 1e-10):

                correlation = np.corrcoef(kl_deviations, coef_changes)[0, 1]

                # Validate correlation result
                if np.isnan(correlation) or np.isinf(correlation):
                    self.current_metrics.coef_adaptation_quality = 0.0
                else:
                    # Good adaptation: positive correlation (coef changes more when KL deviates more)
                    adaptation_quality = max(0, correlation)
                    self.current_metrics.coef_adaptation_quality = min(1.0, adaptation_quality)
            else:
                self.current_metrics.coef_adaptation_quality = 0.0
        except (ValueError, RuntimeError):
            self.current_metrics.coef_adaptation_quality = 0.0

        # Coefficient stability
        try:
            coef_std = np.std(recent_coef, ddof=1)
            if np.isnan(coef_std) or np.isinf(coef_std):
                self.current_metrics.coef_stability = 1.0
            else:
                # Cap stability to reasonable range
                stability = max(0, 1 - min(coef_std, 1.0))
                self.current_metrics.coef_stability = min(1.0, stability)
        except (ValueError, RuntimeError):
            self.current_metrics.coef_stability = 1.0

    def _calculate_health_scores(self):
        """Calculate overall health scores."""
        # KL health score
        kl_health = (
            self.current_metrics.target_range_stability * 0.4 +
            max(0, 1 - self.current_metrics.kl_volatility * 10) * 0.3 +
            max(0, 1 - abs(self.current_metrics.kl_trend) * 10) * 0.3
        )
        self.current_metrics.kl_health_score = max(0, min(1, kl_health))

        # Schedule health score
        schedule_health = (
            self.current_metrics.controller_responsiveness * 0.3 +
            max(0, 1 - self.current_metrics.controller_overshoot * 2) * 0.3 +
            max(0, 1 - self.current_metrics.controller_oscillation * 2) * 0.2 +
            self.current_metrics.coef_adaptation_quality * 0.2
        )
        self.current_metrics.schedule_health_score = max(0, min(1, schedule_health))

    def get_anomalies(self) -> List[Dict[str, Any]]:
        """Detect KL schedule anomalies."""
        anomalies = []

        if len(self.metrics_history) < 10:
            return anomalies

        current = self.current_metrics

        # KL trend anomaly
        if abs(current.kl_trend) > 0.01:  # Strong trend
            anomalies.append({
                "type": "kl_trend_anomaly",
                "severity": "warning" if abs(current.kl_trend) < 0.02 else "critical",
                "message": f"Strong KL trend detected: {current.kl_trend:.4f}",
                "value": current.kl_trend,
                "threshold": 0.01
            })

        # High volatility
        if current.kl_volatility > 0.05:
            anomalies.append({
                "type": "kl_volatility_anomaly",
                "severity": "warning" if current.kl_volatility < 0.1 else "critical",
                "message": f"High KL volatility: {current.kl_volatility:.4f}",
                "value": current.kl_volatility,
                "threshold": 0.05
            })

        # Poor target range performance
        if current.time_in_target_range < 0.5:
            anomalies.append({
                "type": "target_range_anomaly",
                "severity": "warning" if current.time_in_target_range > 0.3 else "critical",
                "message": f"Poor target range performance: {current.time_in_target_range:.2%}",
                "value": current.time_in_target_range,
                "threshold": 0.5
            })

        # Controller issues
        if current.controller_responsiveness < 0.3:
            anomalies.append({
                "type": "controller_responsiveness_anomaly",
                "severity": "warning",
                "message": f"Low controller responsiveness: {current.controller_responsiveness:.3f}",
                "value": current.controller_responsiveness,
                "threshold": 0.3
            })

        if current.controller_overshoot > 0.3:
            anomalies.append({
                "type": "controller_overshoot_anomaly",
                "severity": "warning",
                "message": f"High controller overshoot: {current.controller_overshoot:.3f}",
                "value": current.controller_overshoot,
                "threshold": 0.3
            })

        if current.controller_oscillation > 0.5:
            anomalies.append({
                "type": "controller_oscillation_anomaly",
                "severity": "warning",
                "message": f"High controller oscillation: {current.controller_oscillation:.3f}",
                "value": current.controller_oscillation,
                "threshold": 0.5
            })

        # Coefficient adaptation issues
        if current.coef_adaptation_quality < 0.2:
            anomalies.append({
                "type": "coef_adaptation_anomaly",
                "severity": "warning",
                "message": f"Poor coefficient adaptation: {current.coef_adaptation_quality:.3f}",
                "value": current.coef_adaptation_quality,
                "threshold": 0.2
            })

        return anomalies

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of KL schedule performance."""
        if not self.metrics_history:
            return {}

        return {
            "total_steps": len(self.metrics_history),
            "current_kl": self.current_metrics.current_kl,
            "current_kl_coef": self.current_metrics.current_kl_coef,
            "kl_target": self.kl_target,
            "kl_health_score": self.current_metrics.kl_health_score,
            "schedule_health_score": self.current_metrics.schedule_health_score,
            "time_in_target_range": self.current_metrics.time_in_target_range,
            "target_range_violations": self.current_metrics.target_range_violations,
            "controller_responsiveness": self.current_metrics.controller_responsiveness,
            "controller_overshoot": self.current_metrics.controller_overshoot,
            "controller_oscillation": self.current_metrics.controller_oscillation,
            "coef_adaptation_quality": self.current_metrics.coef_adaptation_quality,
            "anomalies": self.get_anomalies()
        }
