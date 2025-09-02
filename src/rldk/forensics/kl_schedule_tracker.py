"""Comprehensive KL schedule tracking and analysis for PPO training."""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import warnings


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
    
    def update(self, step: int, kl_value: float, kl_coef: float) -> KLScheduleMetrics:
        """Update tracker with new KL and coefficient values."""
        # Store data
        self.kl_history.append(kl_value)
        self.kl_coef_history.append(kl_coef)
        self.step_history.append(step)
        
        # Update current metrics
        self.current_metrics.current_kl = kl_value
        self.current_metrics.current_kl_coef = kl_coef
        
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
        
        # Calculate linear trend
        if len(recent_kl) > 1:
            trend_slope = np.polyfit(steps, recent_kl, 1)[0]
            self.current_metrics.kl_trend = trend_slope
    
    def _analyze_kl_volatility(self):
        """Analyze KL divergence volatility."""
        if len(self.kl_history) < 10:
            return
        
        recent_kl = list(self.kl_history)[-10:]
        self.current_metrics.kl_volatility = float(np.std(recent_kl))
    
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
            kl_changes = np.diff(recent_kl)
            coef_changes = np.diff(recent_coef)
            
            # Calculate correlation between KL changes and coefficient changes
            if len(kl_changes) > 1 and np.std(kl_changes) > 0 and np.std(coef_changes) > 0:
                correlation = np.corrcoef(kl_changes, coef_changes)[0, 1]
                # Good responsiveness: negative correlation (coef increases when KL is high)
                self.current_metrics.controller_responsiveness = max(0, -correlation)
            else:
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
        kl_deviations = [abs(kl - self.kl_target) for kl in recent_kl[:-1]]
        if len(kl_deviations) > 1 and np.std(kl_deviations) > 0 and np.std(coef_changes) > 0:
            correlation = np.corrcoef(kl_deviations, coef_changes)[0, 1]
            # Good adaptation: positive correlation (coef changes more when KL deviates more)
            self.current_metrics.coef_adaptation_quality = max(0, correlation)
        else:
            self.current_metrics.coef_adaptation_quality = 0.0
        
        # Coefficient stability
        coef_std = np.std(recent_coef)
        self.current_metrics.coef_stability = max(0, 1 - coef_std)
    
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