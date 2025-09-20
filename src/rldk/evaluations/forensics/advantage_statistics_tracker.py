"""Comprehensive advantage statistics tracking and analysis for PPO training."""

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class AdvantageStatisticsMetrics:
    """Container for advantage statistics tracking metrics."""

    # Current advantage statistics
    advantage_mean: float = 0.0
    advantage_std: float = 0.0
    advantage_min: float = 0.0
    advantage_max: float = 0.0
    advantage_median: float = 0.0

    # Distribution analysis
    advantage_skewness: float = 0.0
    advantage_kurtosis: float = 0.0
    advantage_percentiles: Dict[str, float] = None

    # Normalization analysis
    advantage_normalization_health: float = 1.0
    advantage_bias: float = 0.0
    advantage_scale_stability: float = 1.0

    # Trend analysis
    advantage_mean_trend: float = 0.0
    advantage_std_trend: float = 0.0
    advantage_volatility: float = 0.0

    # Anomaly detection
    advantage_bias_risk: float = 0.0
    advantage_scale_risk: float = 0.0
    advantage_distribution_risk: float = 0.0

    # Health indicators
    advantage_health_score: float = 1.0
    advantage_quality_score: float = 1.0

    def __post_init__(self):
        if self.advantage_percentiles is None:
            self.advantage_percentiles = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }


class AdvantageStatisticsTracker:
    """Advanced advantage statistics tracking and analysis for PPO training."""

    def __init__(
        self,
        window_size: int = 100,
        trend_window: int = 20,
        bias_threshold: float = 0.1,
        scale_threshold: float = 2.0,
        distribution_window: int = 50,
    ):
        """Initialize advantage statistics tracker.

        Args:
            window_size: Size of rolling window for analysis
            trend_window: Window size for trend analysis
            bias_threshold: Threshold for bias detection
            scale_threshold: Threshold for scale anomaly detection
            distribution_window: Window size for distribution analysis
        """
        self.window_size = window_size
        self.trend_window = trend_window
        self.bias_threshold = bias_threshold
        self.scale_threshold = scale_threshold
        self.distribution_window = distribution_window

        # Data storage
        self.advantage_mean_history: deque = deque(maxlen=window_size)
        self.advantage_std_history: deque = deque(maxlen=window_size)
        self.advantage_min_history: deque = deque(maxlen=window_size)
        self.advantage_max_history: deque = deque(maxlen=window_size)
        self.advantage_median_history: deque = deque(maxlen=window_size)
        self.step_history: deque = deque(maxlen=window_size)

        # Distribution analysis storage
        self.advantage_samples: deque = deque(maxlen=distribution_window * 10)  # Store more samples for distribution

        # Analysis state
        self.current_metrics = AdvantageStatisticsMetrics()
        self.metrics_history: List[AdvantageStatisticsMetrics] = []

        print(f"📈 Advantage Statistics Tracker initialized - "
              f"Bias threshold: {bias_threshold}, Scale threshold: {scale_threshold}")

    def update(
        self,
        step: int,
        advantage_mean: float,
        advantage_std: float,
        advantage_min: Optional[float] = None,
        advantage_max: Optional[float] = None,
        advantage_median: Optional[float] = None,
        advantage_samples: Optional[List[float]] = None
    ) -> AdvantageStatisticsMetrics:
        """Update tracker with new advantage statistics."""
        # Store basic statistics
        self.advantage_mean_history.append(advantage_mean)
        self.advantage_std_history.append(advantage_std)
        self.advantage_min_history.append(advantage_min or 0.0)
        self.advantage_max_history.append(advantage_max or 0.0)
        self.advantage_median_history.append(advantage_median or advantage_mean)
        self.step_history.append(step)

        # Store samples for distribution analysis
        if advantage_samples:
            self.advantage_samples.extend(advantage_samples)

        # Update current metrics
        self.current_metrics.advantage_mean = advantage_mean
        self.current_metrics.advantage_std = advantage_std
        self.current_metrics.advantage_min = advantage_min or 0.0
        self.current_metrics.advantage_max = advantage_max or 0.0
        self.current_metrics.advantage_median = advantage_median or advantage_mean

        # Perform analysis if we have enough data
        if len(self.advantage_mean_history) >= 2:
            self._analyze_advantage_trends()
            self._analyze_advantage_volatility()
            self._analyze_advantage_normalization()
            self._analyze_advantage_distribution()
            self._detect_advantage_anomalies()
            self._calculate_health_scores()

        # Store metrics
        metrics_copy = AdvantageStatisticsMetrics(**self.current_metrics.to_dict())
        self.metrics_history.append(metrics_copy)

        return metrics_copy

    def _analyze_advantage_trends(self):
        """Analyze advantage statistics trends."""
        if len(self.advantage_mean_history) < self.trend_window:
            return

        # Mean trend
        recent_mean = list(self.advantage_mean_history)[-self.trend_window:]
        steps = list(range(len(recent_mean)))
        if len(recent_mean) > 1:
            self.current_metrics.advantage_mean_trend = np.polyfit(steps, recent_mean, 1)[0]

        # Std trend
        recent_std = list(self.advantage_std_history)[-self.trend_window:]
        if len(recent_std) > 1:
            self.current_metrics.advantage_std_trend = np.polyfit(steps, recent_std, 1)[0]

    def _analyze_advantage_volatility(self):
        """Analyze advantage volatility."""
        if len(self.advantage_mean_history) < 10:
            return

        recent_mean = list(self.advantage_mean_history)[-10:]
        self.current_metrics.advantage_volatility = float(np.std(recent_mean))

    def _analyze_advantage_normalization(self):
        """Analyze advantage normalization health."""
        if len(self.advantage_mean_history) < 10:
            return

        recent_mean = list(self.advantage_mean_history)[-10:]
        recent_std = list(self.advantage_std_history)[-10:]

        # Bias analysis: mean should be close to 0
        mean_bias = abs(np.mean(recent_mean))
        self.current_metrics.advantage_bias = mean_bias

        # Scale stability: std should be relatively stable
        std_cv = np.std(recent_std) / max(np.mean(recent_std), 1e-8)
        self.current_metrics.advantage_scale_stability = max(0, 1 - std_cv)

        # Overall normalization health
        bias_health = max(0, 1 - mean_bias / self.bias_threshold)
        scale_health = self.current_metrics.advantage_scale_stability

        self.current_metrics.advantage_normalization_health = (bias_health + scale_health) / 2

    def _analyze_advantage_distribution(self):
        """Analyze advantage distribution characteristics."""
        if len(self.advantage_samples) < 50:
            return

        # Get recent samples
        recent_samples = list(self.advantage_samples)[-50:]
        samples_array = np.array(recent_samples)

        # Calculate distribution statistics
        self.current_metrics.advantage_skewness = float(self._calculate_skewness(samples_array))
        self.current_metrics.advantage_kurtosis = float(self._calculate_kurtosis(samples_array))

        # Calculate percentiles
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        for p in percentiles:
            self.current_metrics.advantage_percentiles[f"p{p}"] = float(np.percentile(samples_array, p))

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        if len(data) < 3:
            return 0.0

        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0

        skewness = np.mean(((data - mean) / std) ** 3)
        return skewness

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        if len(data) < 4:
            return 0.0

        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0

        kurtosis = np.mean(((data - mean) / std) ** 4) - 3  # Excess kurtosis
        return kurtosis

    def _detect_advantage_anomalies(self):
        """Detect advantage anomalies."""
        if len(self.advantage_mean_history) < 5:
            return

        recent_mean = list(self.advantage_mean_history)[-5:]
        recent_std = list(self.advantage_std_history)[-5:]

        # Bias risk
        mean_bias = abs(np.mean(recent_mean))
        self.current_metrics.advantage_bias_risk = min(1.0, mean_bias / self.bias_threshold)

        # Scale risk
        std_cv = np.std(recent_std) / max(np.mean(recent_std), 1e-8)
        self.current_metrics.advantage_scale_risk = min(1.0, std_cv / self.scale_threshold)

        # Distribution risk (based on skewness and kurtosis)
        if len(self.advantage_samples) >= 20:
            recent_samples = list(self.advantage_samples)[-20:]
            samples_array = np.array(recent_samples)

            skewness = abs(self._calculate_skewness(samples_array))
            kurtosis = abs(self._calculate_kurtosis(samples_array))

            # High skewness or kurtosis indicates distribution issues
            distribution_risk = min(1.0, (skewness + kurtosis) / 4.0)
            self.current_metrics.advantage_distribution_risk = distribution_risk

    def _calculate_health_scores(self):
        """Calculate overall health scores."""
        # Advantage health score
        advantage_health = (
            self.current_metrics.advantage_normalization_health * 0.4 +
            max(0, 1 - self.current_metrics.advantage_bias_risk) * 0.3 +
            max(0, 1 - self.current_metrics.advantage_scale_risk) * 0.2 +
            max(0, 1 - self.current_metrics.advantage_distribution_risk) * 0.1
        )
        self.current_metrics.advantage_health_score = max(0, min(1, advantage_health))

        # Advantage quality score
        quality_score = (
            self.current_metrics.advantage_scale_stability * 0.3 +
            max(0, 1 - abs(self.current_metrics.advantage_mean_trend) * 10) * 0.3 +
            max(0, 1 - self.current_metrics.advantage_volatility * 5) * 0.2 +
            max(0, 1 - abs(self.current_metrics.advantage_skewness) / 2) * 0.2
        )
        self.current_metrics.advantage_quality_score = max(0, min(1, quality_score))

    def get_anomalies(self) -> List[Dict[str, Any]]:
        """Detect advantage anomalies."""
        anomalies = []

        if len(self.metrics_history) < 5:
            return anomalies

        current = self.current_metrics

        # Bias anomaly
        if current.advantage_bias_risk > 0.5:
            anomalies.append({
                "type": "advantage_bias_anomaly",
                "severity": "critical" if current.advantage_bias_risk > 0.8 else "warning",
                "message": f"High advantage bias: {current.advantage_bias:.4f}",
                "value": current.advantage_bias,
                "threshold": self.bias_threshold
            })

        # Scale anomaly
        if current.advantage_scale_risk > 0.5:
            anomalies.append({
                "type": "advantage_scale_anomaly",
                "severity": "warning",
                "message": f"High advantage scale risk: {current.advantage_scale_risk:.3f}",
                "value": current.advantage_scale_risk,
                "threshold": 0.5
            })

        # Distribution anomaly
        if current.advantage_distribution_risk > 0.5:
            anomalies.append({
                "type": "advantage_distribution_anomaly",
                "severity": "warning",
                "message": f"High advantage distribution risk: {current.advantage_distribution_risk:.3f}",
                "value": current.advantage_distribution_risk,
                "threshold": 0.5
            })

        # Poor normalization health
        if current.advantage_normalization_health < 0.5:
            anomalies.append({
                "type": "advantage_normalization_anomaly",
                "severity": "warning",
                "message": f"Poor advantage normalization: {current.advantage_normalization_health:.3f}",
                "value": current.advantage_normalization_health,
                "threshold": 0.5
            })

        # Strong mean trend
        if abs(current.advantage_mean_trend) > 0.01:
            anomalies.append({
                "type": "advantage_trend_anomaly",
                "severity": "warning",
                "message": f"Strong advantage mean trend: {current.advantage_mean_trend:.4f}",
                "value": current.advantage_mean_trend,
                "threshold": 0.01
            })

        # High volatility
        if current.advantage_volatility > 0.1:
            anomalies.append({
                "type": "advantage_volatility_anomaly",
                "severity": "warning",
                "message": f"High advantage volatility: {current.advantage_volatility:.4f}",
                "value": current.advantage_volatility,
                "threshold": 0.1
            })

        # Extreme skewness
        if abs(current.advantage_skewness) > 2.0:
            anomalies.append({
                "type": "advantage_skewness_anomaly",
                "severity": "warning",
                "message": f"Extreme advantage skewness: {current.advantage_skewness:.3f}",
                "value": current.advantage_skewness,
                "threshold": 2.0
            })

        # Extreme kurtosis
        if abs(current.advantage_kurtosis) > 3.0:
            anomalies.append({
                "type": "advantage_kurtosis_anomaly",
                "severity": "warning",
                "message": f"Extreme advantage kurtosis: {current.advantage_kurtosis:.3f}",
                "value": current.advantage_kurtosis,
                "threshold": 3.0
            })

        return anomalies

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of advantage statistics analysis."""
        if not self.metrics_history:
            return {}

        return {
            "total_steps": len(self.metrics_history),
            "current_advantage_mean": self.current_metrics.advantage_mean,
            "current_advantage_std": self.current_metrics.advantage_std,
            "current_advantage_min": self.current_metrics.advantage_min,
            "current_advantage_max": self.current_metrics.advantage_max,
            "current_advantage_median": self.current_metrics.advantage_median,
            "advantage_health_score": self.current_metrics.advantage_health_score,
            "advantage_quality_score": self.current_metrics.advantage_quality_score,
            "advantage_normalization_health": self.current_metrics.advantage_normalization_health,
            "advantage_bias": self.current_metrics.advantage_bias,
            "advantage_scale_stability": self.current_metrics.advantage_scale_stability,
            "advantage_volatility": self.current_metrics.advantage_volatility,
            "advantage_skewness": self.current_metrics.advantage_skewness,
            "advantage_kurtosis": self.current_metrics.advantage_kurtosis,
            "advantage_bias_risk": self.current_metrics.advantage_bias_risk,
            "advantage_scale_risk": self.current_metrics.advantage_scale_risk,
            "advantage_distribution_risk": self.current_metrics.advantage_distribution_risk,
            "advantage_mean_trend": self.current_metrics.advantage_mean_trend,
            "advantage_std_trend": self.current_metrics.advantage_std_trend,
            "advantage_percentiles": self.current_metrics.advantage_percentiles,
            "anomalies": self.get_anomalies()
        }
