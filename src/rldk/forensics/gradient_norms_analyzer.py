"""Comprehensive gradient norms analysis for PPO training.

Enhanced with advanced detection methods to catch subtle gradient instabilities
that traditional threshold-based approaches might miss:

- Adaptive threshold detection based on historical patterns
- Statistical outlier detection using Z-score and IQR methods
- Momentum analysis for detecting accelerating instabilities
- Multi-scale analysis across different time windows
- Early warning system for gradual deterioration
- Exponential smoothing and change point detection

This addresses the limitation of fixed thresholds that can miss subtle
gradient issues leading to training instabilities.
"""

import math
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# Optimized mathematical functions with validation
def polyfit(x, y, degree):
    """
    Linear regression implementation with validation.

    Returns [intercept, slope] for degree=1 linear fitting.
    Validates input and handles edge cases for numerical stability.
    """
    if degree != 1 or len(x) != len(y) or len(x) < 2:
        return [0.0, 0.0]

    # Input validation
    if not all(isinstance(val, (int, float)) for val in x + y):
        raise ValueError("All values must be numeric")

    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(x[i] * y[i] for i in range(n))
    sum_x2 = sum(x[i] ** 2 for i in range(n))

    denominator = n * sum_x2 - sum_x * sum_x

    # Numerical stability check
    if abs(denominator) < 1e-10:
        return [0.0, sum_y / n]

    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n

    # Bounds checking for extreme values
    if abs(slope) > 1e6 or abs(intercept) > 1e6:
        return [0.0, sum_y / n]

    return [intercept, slope]  # Fixed: numpy compatibility order

def _validate_math_functions():
    """
    Simple validation of mathematical functions against known test cases.
    Called during initialization to ensure accuracy.
    """
    # Test polyfit with simple linear data
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]  # Perfect linear relationship: y = 2x
    intercept, slope = polyfit(x, y, 1)  # Fixed: correct order [intercept, slope]

    # Allow small floating point errors
    if abs(slope - 2.0) > 1e-10 or abs(intercept - 0.0) > 1e-10:
        warnings.warn("polyfit function may have accuracy issues", UserWarning, stacklevel=2)

    # Test mean and std with known values
    test_values = [1, 2, 3, 4, 5]
    expected_mean = 3.0
    expected_std = 1.5811388300841898  # sqrt(2.5) - corrected for sample std (n-1)

    if abs(mean(test_values) - expected_mean) > 1e-10:
        warnings.warn("mean function may have accuracy issues", UserWarning, stacklevel=2)

    if abs(std(test_values) - expected_std) > 1e-8:
        warnings.warn("std function may have accuracy issues", UserWarning, stacklevel=2)

def mean(values):
    """Calculate mean."""
    return sum(values) / len(values) if values else 0.0

def std(values):
    """Calculate sample standard deviation (using n-1 denominator)."""
    if not values or len(values) < 2:
        return 0.0
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)  # Fixed: use n-1
    return math.sqrt(variance)

def percentile(values, p):
    """Calculate percentile with proper error handling."""
    if not values:
        return 0.0
    if not (0 <= p <= 100):
        raise ValueError("Percentile must be between 0 and 100")

    sorted_values = sorted(values)
    n = len(sorted_values)

    # Use linear interpolation for more accurate percentile calculation
    if p == 0:
        return sorted_values[0]
    elif p == 100:
        return sorted_values[-1]

    # Calculate index with proper interpolation
    index = (p / 100) * (n - 1)
    lower_idx = int(index)
    upper_idx = min(lower_idx + 1, n - 1)

    if lower_idx == upper_idx:
        return sorted_values[lower_idx]

    # Linear interpolation between adjacent values
    weight = index - lower_idx
    return sorted_values[lower_idx] * (1 - weight) + sorted_values[upper_idx] * weight

def var(values):
    """Calculate variance."""
    if not values or len(values) < 2:
        return 0.0
    m = mean(values)
    return sum((x - m) ** 2 for x in values) / len(values)

def zscore(values):
    """Calculate z-scores with proper error handling."""
    if not values or len(values) < 2:
        return []

    m = mean(values)
    s = std(values)

    # Handle case where all values are identical (std = 0)
    if s < 1e-10:
        return [0.0] * len(values)

    # Calculate z-scores with bounds checking
    z_scores = []
    for x in values:
        if not isinstance(x, (int, float)) or not math.isfinite(x):
            z_scores.append(0.0)  # Handle non-numeric or infinite values
        else:
            z_score = (x - m) / s
            # Bound extreme z-scores to prevent overflow
            z_scores.append(max(-10.0, min(10.0, z_score)))

    return z_scores


@dataclass
class GradientNormsMetrics:
    """Container for gradient norms analysis metrics."""

    # Current gradient norms
    policy_grad_norm: float = 0.0
    value_grad_norm: float = 0.0
    total_grad_norm: float = 0.0

    # Gradient ratios and relationships
    policy_value_ratio: float = 0.0
    policy_total_ratio: float = 0.0
    value_total_ratio: float = 0.0

    # Gradient flow analysis
    gradient_flow_health: float = 1.0
    gradient_balance: float = 1.0
    gradient_stability: float = 1.0

    # Trend analysis
    policy_grad_trend: float = 0.0
    value_grad_trend: float = 0.0
    ratio_trend: float = 0.0

    # Anomaly detection
    exploding_gradient_risk: float = 0.0
    vanishing_gradient_risk: float = 0.0
    gradient_imbalance_risk: float = 0.0

    # Enhanced detection metrics
    adaptive_explosion_risk: float = 0.0
    statistical_outlier_risk: float = 0.0
    momentum_instability_risk: float = 0.0
    multi_scale_anomaly_risk: float = 0.0
    early_warning_score: float = 0.0

    # Advanced trend analysis
    exponential_smoothing_trend: float = 0.0
    change_point_detected: bool = False
    gradient_acceleration: float = 0.0

    # Health indicators
    gradient_health_score: float = 1.0
    training_stability: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }


class GradientNormsAnalyzer:
    """Advanced gradient norms analysis for PPO training."""

    def __init__(
        self,
        window_size: int = 50,
        trend_window: int = 20,
        exploding_threshold: float = 10.0,
        vanishing_threshold: float = 0.001,
        imbalance_threshold: float = 0.1,
        # Enhanced detection parameters with justified defaults
        z_score_threshold: float = 2.5,        # 2.5σ covers ~98.8% of normal distribution
        iqr_multiplier: float = 1.5,           # Standard IQR outlier detection multiplier
        momentum_window: int = 10,              # Balance between responsiveness and stability
        early_warning_threshold: float = 0.7,  # 70% confidence threshold for warnings
        smoothing_alpha: float = 0.3,          # Exponential smoothing: 0.3 = moderate smoothing
        change_point_sensitivity: float = 0.1, # 10% change relative to noise level
        enable_advanced_detection: bool = True, # Toggle for performance vs accuracy trade-off
    ):
        """Initialize gradient norms analyzer.

        Args:
            window_size: Size of rolling window for analysis
            trend_window: Window size for trend analysis
            exploding_threshold: Threshold for exploding gradient detection
            vanishing_threshold: Threshold for vanishing gradient detection
            imbalance_threshold: Threshold for gradient imbalance detection
            z_score_threshold: Z-score threshold for statistical outlier detection (2.5σ = 98.8% normal)
            iqr_multiplier: IQR multiplier for outlier detection (1.5 = standard Tukey method)
            momentum_window: Window size for momentum analysis (10 = balance of responsiveness/stability)
            early_warning_threshold: Threshold for early warning system (0.7 = 70% confidence)
            smoothing_alpha: Alpha parameter for exponential smoothing (0.3 = moderate smoothing)
            change_point_sensitivity: Sensitivity for change point detection (0.1 = 10% change threshold)
            enable_advanced_detection: Enable/disable advanced detection for performance tuning
        """
        self.window_size = window_size
        self.trend_window = trend_window
        self.exploding_threshold = exploding_threshold
        self.vanishing_threshold = vanishing_threshold
        self.imbalance_threshold = imbalance_threshold

        # Enhanced detection parameters
        self.z_score_threshold = z_score_threshold
        self.iqr_multiplier = iqr_multiplier
        self.momentum_window = momentum_window
        self.early_warning_threshold = early_warning_threshold
        self.smoothing_alpha = smoothing_alpha
        self.change_point_sensitivity = change_point_sensitivity
        self.enable_advanced_detection = enable_advanced_detection

        # Data storage
        self.policy_grad_history: deque = deque(maxlen=window_size)
        self.value_grad_history: deque = deque(maxlen=window_size)
        self.total_grad_history: deque = deque(maxlen=window_size)
        self.ratio_history: deque = deque(maxlen=window_size)
        self.step_history: deque = deque(maxlen=window_size)

        # Enhanced analysis storage
        self.exponential_smoothed_history: deque = deque(maxlen=window_size)
        self.momentum_history: deque = deque(maxlen=momentum_window)
        self.adaptive_thresholds: Dict[str, float] = {}
        self.baseline_established = False

        # Analysis state
        self.current_metrics = GradientNormsMetrics()
        self.metrics_history: List[GradientNormsMetrics] = []

        # Validate mathematical functions on initialization
        _validate_math_functions()

        print(f"📊 Enhanced Gradient Norms Analyzer initialized - "
              f"Exploding: {exploding_threshold}, Vanishing: {vanishing_threshold}, "
              f"Imbalance: {imbalance_threshold}, Z-score: {z_score_threshold}, "
              f"Advanced: {enable_advanced_detection}")

    def update(
        self,
        step: int,
        policy_grad_norm: float,
        value_grad_norm: float,
        total_grad_norm: Optional[float] = None
    ) -> GradientNormsMetrics:
        """Update analyzer with new gradient norm values."""
        # Calculate total gradient norm if not provided
        if total_grad_norm is None:
            total_grad_norm = (policy_grad_norm**2 + value_grad_norm**2)**0.5

        # Store data
        self.policy_grad_history.append(policy_grad_norm)
        self.value_grad_history.append(value_grad_norm)
        self.total_grad_history.append(total_grad_norm)
        self.step_history.append(step)

        # Calculate ratios
        policy_value_ratio = policy_grad_norm / max(value_grad_norm, 1e-8)
        policy_total_ratio = policy_grad_norm / max(total_grad_norm, 1e-8)
        value_total_ratio = value_grad_norm / max(total_grad_norm, 1e-8)

        self.ratio_history.append(policy_value_ratio)

        # Update current metrics
        self.current_metrics.policy_grad_norm = policy_grad_norm
        self.current_metrics.value_grad_norm = value_grad_norm
        self.current_metrics.total_grad_norm = total_grad_norm
        self.current_metrics.policy_value_ratio = policy_value_ratio
        self.current_metrics.policy_total_ratio = policy_total_ratio
        self.current_metrics.value_total_ratio = value_total_ratio

        # Perform analysis if we have enough data
        if len(self.policy_grad_history) >= 2:
            self._analyze_gradient_trends()
            self._analyze_gradient_flow_health()
            self._analyze_gradient_balance()
            self._analyze_gradient_stability()
            self._detect_gradient_anomalies()

            # Enhanced detection methods (conditionally enabled for performance)
            if self.enable_advanced_detection:
                self._adaptive_threshold_detection()
                self._statistical_outlier_detection()
                self._momentum_analysis()
                self._multi_scale_analysis()
                self._early_warning_system()
                self._exponential_smoothing_analysis()
                self._change_point_detection()

            self._calculate_health_scores()

        # Store metrics
        metrics_copy = GradientNormsMetrics(**self.current_metrics.to_dict())
        self.metrics_history.append(metrics_copy)

        return metrics_copy

    def _analyze_gradient_trends(self):
        """Analyze gradient norm trends."""
        if len(self.policy_grad_history) < self.trend_window:
            return

        # Policy gradient trend
        recent_policy = self._get_recent_values(self.policy_grad_history, self.trend_window)
        steps = list(range(len(recent_policy)))
        if len(recent_policy) > 1:
            _, slope = polyfit(steps, recent_policy, 1)  # Fixed: use correct order
            self.current_metrics.policy_grad_trend = slope

        # Value gradient trend
        recent_value = self._get_recent_values(self.value_grad_history, self.trend_window)
        if len(recent_value) > 1:
            _, slope = polyfit(steps, recent_value, 1)  # Fixed: use correct order
            self.current_metrics.value_grad_trend = slope

        # Ratio trend
        recent_ratio = self._get_recent_values(self.ratio_history, self.trend_window)
        if len(recent_ratio) > 1:
            _, slope = polyfit(steps, recent_ratio, 1)  # Fixed: use correct order
            self.current_metrics.ratio_trend = slope

    def _analyze_gradient_flow_health(self):
        """Analyze gradient flow health."""
        if len(self.total_grad_history) < 10:
            return

        recent_total = list(self.total_grad_history)[-10:]

        # Healthy gradient norms are typically between 0.1 and 10
        healthy_count = sum(1 for norm in recent_total
                          if 0.1 <= norm <= 10.0)

        self.current_metrics.gradient_flow_health = healthy_count / len(recent_total)

    def _analyze_gradient_balance(self):
        """Analyze balance between policy and value gradients."""
        if len(self.ratio_history) < 10:
            return

        recent_ratios = list(self.ratio_history)[-10:]

        # Ideal ratio is around 1.0 (balanced)
        # Calculate how close ratios are to 1.0
        balance_scores = [max(0, 1 - abs(ratio - 1.0)) for ratio in recent_ratios]
        self.current_metrics.gradient_balance = mean(balance_scores)

    def _analyze_gradient_stability(self):
        """Analyze gradient stability over time."""
        if len(self.policy_grad_history) < 10:
            return

        recent_policy = list(self.policy_grad_history)[-10:]
        recent_value = list(self.value_grad_history)[-10:]

        # Calculate coefficient of variation (std/mean) for stability
        policy_cv = std(recent_policy) / max(mean(recent_policy), 1e-8)
        value_cv = std(recent_value) / max(mean(recent_value), 1e-8)

        # Lower CV is more stable
        policy_stability = max(0, 1 - policy_cv)
        value_stability = max(0, 1 - value_cv)

        self.current_metrics.gradient_stability = (policy_stability + value_stability) / 2

    def _detect_gradient_anomalies(self):
        """Detect gradient anomalies."""
        if len(self.policy_grad_history) < 5:
            return

        list(self.policy_grad_history)[-5:]
        list(self.value_grad_history)[-5:]
        recent_total = list(self.total_grad_history)[-5:]
        recent_ratios = list(self.ratio_history)[-5:]

        # Exploding gradient risk
        exploding_count = sum(1 for norm in recent_total if norm > self.exploding_threshold)
        self.current_metrics.exploding_gradient_risk = exploding_count / len(recent_total)

        # Vanishing gradient risk
        vanishing_count = sum(1 for norm in recent_total if norm < self.vanishing_threshold)
        self.current_metrics.vanishing_gradient_risk = vanishing_count / len(recent_total)

        # Gradient imbalance risk
        imbalance_count = sum(1 for ratio in recent_ratios
                            if ratio < self.imbalance_threshold or ratio > 1/self.imbalance_threshold)
        self.current_metrics.gradient_imbalance_risk = imbalance_count / len(recent_ratios)

    def _calculate_health_scores(self):
        """Calculate overall health scores incorporating enhanced detection metrics."""
        # Enhanced gradient health score
        gradient_health = (
            self.current_metrics.gradient_flow_health * 0.2 +
            self.current_metrics.gradient_balance * 0.2 +
            self.current_metrics.gradient_stability * 0.15 +
            max(0, 1 - self.current_metrics.exploding_gradient_risk) * 0.1 +
            max(0, 1 - self.current_metrics.vanishing_gradient_risk) * 0.1 +
            max(0, 1 - self.current_metrics.adaptive_explosion_risk) * 0.1 +
            max(0, 1 - self.current_metrics.statistical_outlier_risk) * 0.1 +
            max(0, 1 - self.current_metrics.momentum_instability_risk) * 0.05
        )
        self.current_metrics.gradient_health_score = max(0, min(1, gradient_health))

        # Enhanced training stability
        training_stability = (
            self.current_metrics.gradient_stability * 0.25 +
            max(0, 1 - self.current_metrics.gradient_imbalance_risk) * 0.2 +
            max(0, 1 - abs(self.current_metrics.ratio_trend) * 10) * 0.15 +
            max(0, 1 - self.current_metrics.multi_scale_anomaly_risk) * 0.2 +
            max(0, 1 - self.current_metrics.early_warning_score) * 0.2
        )
        self.current_metrics.training_stability = max(0, min(1, training_stability))

    def _get_recent_values(self, history, count):
        """Efficiently get recent values from a deque without full list conversion."""
        if len(history) <= count:
            return list(history)
        return [history[i] for i in range(len(history) - count, len(history))]

    def _adaptive_threshold_detection(self):
        """
        Optimized adaptive threshold detection with caching.

        Uses historical patterns to dynamically adjust detection thresholds,
        reducing false positives while maintaining sensitivity.
        """
        if len(self.total_grad_history) < 20:
            return

        # Establish baseline if not done yet - FIXED: remove redundant condition
        if not self.baseline_established:
            self._establish_baseline()

        # Use only recent data for efficiency
        recent_norms = self._get_recent_values(self.total_grad_history, 10)

        # Cache threshold calculations - FIXED: handle None cache properly
        if not hasattr(self, '_cached_threshold') or self._cached_threshold is None:
            baseline_mean = self.adaptive_thresholds.get('mean', 1.0)
            baseline_std = self.adaptive_thresholds.get('std', 0.5)
            baseline_p95 = self.adaptive_thresholds.get('p95', 5.0)

            # Adaptive explosion threshold: mean + 3*std or 95th percentile * 2
            self._cached_threshold = max(baseline_mean + 3 * baseline_std, baseline_p95 * 2)

        # Count violations efficiently
        explosion_count = sum(1 for norm in recent_norms if norm > self._cached_threshold)
        self.current_metrics.adaptive_explosion_risk = explosion_count / len(recent_norms)

        # Update thresholds less frequently for performance
        if len(self.total_grad_history) % 100 == 0:  # Reduced frequency
            self._update_adaptive_thresholds()
            self._cached_threshold = None  # Invalidate cache - will be rebuilt on next call

    def _establish_baseline(self):
        """Establish baseline statistics for adaptive threshold detection."""
        if len(self.total_grad_history) < 20:
            return

        baseline_data = list(self.total_grad_history)[:20]
        self.adaptive_thresholds = {
            'mean': mean(baseline_data),
            'std': std(baseline_data),
            'median': sorted(baseline_data)[len(baseline_data)//2],
            'p95': percentile(baseline_data, 95),
            'p99': percentile(baseline_data, 99),
        }
        self.baseline_established = True

    def _update_adaptive_thresholds(self):
        """Update adaptive thresholds based on recent history."""
        if len(self.total_grad_history) < 20:
            return

        # Use last 30% of data for updating thresholds
        update_size = max(10, len(self.total_grad_history) // 3)
        recent_data = list(self.total_grad_history)[-update_size:]

        # Exponential moving average update
        alpha = 0.1
        self.adaptive_thresholds['mean'] = (
            alpha * mean(recent_data) + (1 - alpha) * self.adaptive_thresholds['mean']
        )
        self.adaptive_thresholds['std'] = (
            alpha * std(recent_data) + (1 - alpha) * self.adaptive_thresholds['std']
        )
        self.adaptive_thresholds['p95'] = (
            alpha * percentile(recent_data, 95) + (1 - alpha) * self.adaptive_thresholds['p95']
        )

    def _statistical_outlier_detection(self):
        """Detect outliers using statistical methods (Z-score and IQR)."""
        if len(self.total_grad_history) < 10:
            return

        recent_norms = list(self.total_grad_history)[-10:]

        # Z-score based outlier detection
        if std(recent_norms) > 1e-8:
            z_scores = [abs(z) for z in zscore(recent_norms)]
            outlier_count = sum(1 for z in z_scores if z > self.z_score_threshold)
            z_score_risk = outlier_count / len(recent_norms)
        else:
            z_score_risk = 0.0

        # IQR based outlier detection
        q1 = percentile(recent_norms, 25)
        q3 = percentile(recent_norms, 75)
        iqr = q3 - q1
        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr

        iqr_outlier_count = sum(1 for norm in recent_norms if norm < lower_bound or norm > upper_bound)
        iqr_risk = iqr_outlier_count / len(recent_norms)

        # Combined statistical outlier risk
        self.current_metrics.statistical_outlier_risk = max(z_score_risk, iqr_risk)

    def _momentum_analysis(self):
        """Analyze gradient momentum to detect accelerating instabilities."""
        if len(self.total_grad_history) < self.momentum_window:
            return

        recent_norms = list(self.total_grad_history)[-self.momentum_window:]

        # Calculate momentum (rate of change)
        momentum_values = []
        for i in range(1, len(recent_norms)):
            momentum = recent_norms[i] - recent_norms[i-1]
            momentum_values.append(momentum)

        if len(momentum_values) < 3:
            return

        # Store momentum for trend analysis
        self.momentum_history.append(momentum_values[-1])

        # Calculate gradient acceleration (second derivative)
        acceleration_values = []
        for i in range(1, len(momentum_values)):
            acceleration = momentum_values[i] - momentum_values[i-1]
            acceleration_values.append(acceleration)

        if acceleration_values:
            self.current_metrics.gradient_acceleration = mean(acceleration_values)

            # Detect accelerating instability
            positive_acceleration = sum(1 for acc in acceleration_values if acc > 0)
            acceleration_instability = positive_acceleration / len(acceleration_values)

            # High momentum with positive acceleration indicates instability
            avg_momentum = mean([abs(m) for m in momentum_values])
            momentum_instability = min(1.0, avg_momentum * acceleration_instability)

            self.current_metrics.momentum_instability_risk = momentum_instability

    def _multi_scale_analysis(self):
        """
        Optimized multi-scale analysis with configurable scales.

        Analyzes gradient variance at different time scales to detect
        various types of instabilities efficiently.
        """
        if len(self.total_grad_history) < 10:
            return

        recent_norms = list(self.total_grad_history)

        # Predefined scale configurations for efficiency
        scales = [
            (5, 0.5, 0.5),   # short-term: 5 steps, variance threshold, weight
            (15, 0.3, 0.3),  # medium-term: 15 steps, variance threshold, weight
            (30, 0.2, 0.2)   # long-term: 30 steps, variance threshold, weight
        ]

        total_risk = 0.0
        total_weight = 0.0

        for window_size, variance_threshold, weight in scales:
            if len(recent_norms) >= window_size:
                scale_data = recent_norms[-window_size:]
                if len(scale_data) > 1:
                    scale_var = var(scale_data)
                    scale_mean = mean(scale_data)

                    # Normalize variance relative to mean for scale-invariant detection
                    if scale_mean > 1e-8:
                        normalized_var = scale_var / (scale_mean ** 2)
                        anomaly_risk = min(1.0, normalized_var / variance_threshold)
                        total_risk += anomaly_risk * weight
                        total_weight += weight

        # Normalize by total weight to get final risk score
        self.current_metrics.multi_scale_anomaly_risk = (
            total_risk / total_weight if total_weight > 0 else 0.0
        )

    def _early_warning_system(self):
        """Early warning system for gradual gradient deterioration."""
        if len(self.metrics_history) < 10:
            return

        # Get recent health metrics
        recent_health_scores = [m.gradient_health_score for m in self.metrics_history[-10:]]

        # Calculate trend in health scores
        if len(recent_health_scores) > 2:
            _, health_trend = polyfit(list(range(len(recent_health_scores))), recent_health_scores, 1)  # Fixed: use correct order

            # Early warning if health is declining
            declining_health = max(0, -health_trend)

            # Combine with current stability metrics
            stability_factor = 1 - self.current_metrics.gradient_stability

            # Early warning score (higher = more concerning)
            early_warning = (declining_health * 0.6 + stability_factor * 0.4)
            self.current_metrics.early_warning_score = min(1.0, early_warning)

    def _exponential_smoothing_analysis(self):
        """Apply exponential smoothing for trend analysis."""
        if len(self.total_grad_history) < 3:
            return

        recent_norms = list(self.total_grad_history)
        current_norm = recent_norms[-1]

        # Calculate exponentially smoothed value
        if not self.exponential_smoothed_history:
            smoothed_value = current_norm
        else:
            prev_smoothed = self.exponential_smoothed_history[-1]
            smoothed_value = self.smoothing_alpha * current_norm + (1 - self.smoothing_alpha) * prev_smoothed

        self.exponential_smoothed_history.append(smoothed_value)

        # Calculate smoothed trend
        if len(self.exponential_smoothed_history) >= 3:
            smoothed_values = list(self.exponential_smoothed_history)[-5:]
            if len(smoothed_values) > 1:
                _, slope = polyfit(
                    list(range(len(smoothed_values))), smoothed_values, 1
                )  # Fixed: use correct order
                self.current_metrics.exponential_smoothing_trend = slope

    def _change_point_detection(self):
        """Detect sudden changes in gradient patterns."""
        if len(self.total_grad_history) < 15:
            return

        recent_norms = list(self.total_grad_history)[-15:]

        # Simple change point detection using CUSUM-like approach
        # Calculate mean before and after potential change point
        mid_point = len(recent_norms) // 2

        before_mean = mean(recent_norms[:mid_point])
        after_mean = mean(recent_norms[mid_point:])

        # Calculate change magnitude
        change_magnitude = abs(after_mean - before_mean)
        baseline_noise = std(recent_norms)

        # Detect significant change
        if baseline_noise > 1e-8:
            change_ratio = change_magnitude / baseline_noise
            self.current_metrics.change_point_detected = change_ratio > self.change_point_sensitivity

    def get_anomalies(self) -> List[Dict[str, Any]]:
        """Detect gradient norm anomalies."""
        anomalies = []

        if len(self.metrics_history) < 5:
            return anomalies

        current = self.current_metrics

        # Exploding gradient anomaly
        if current.exploding_gradient_risk > 0.2:
            anomalies.append({
                "type": "exploding_gradient_anomaly",
                "severity": "critical" if current.exploding_gradient_risk > 0.5 else "warning",
                "message": f"Exploding gradient risk: {current.exploding_gradient_risk:.2%}",
                "value": current.exploding_gradient_risk,
                "threshold": 0.2
            })

        # Vanishing gradient anomaly
        if current.vanishing_gradient_risk > 0.2:
            anomalies.append({
                "type": "vanishing_gradient_anomaly",
                "severity": "critical" if current.vanishing_gradient_risk > 0.5 else "warning",
                "message": f"Vanishing gradient risk: {current.vanishing_gradient_risk:.2%}",
                "value": current.vanishing_gradient_risk,
                "threshold": 0.2
            })

        # Gradient imbalance anomaly
        if current.gradient_imbalance_risk > 0.3:
            anomalies.append({
                "type": "gradient_imbalance_anomaly",
                "severity": "warning",
                "message": f"Gradient imbalance risk: {current.gradient_imbalance_risk:.2%}",
                "value": current.gradient_imbalance_risk,
                "threshold": 0.3
            })

        # Poor gradient flow health
        if current.gradient_flow_health < 0.5:
            anomalies.append({
                "type": "gradient_flow_anomaly",
                "severity": "warning",
                "message": f"Poor gradient flow health: {current.gradient_flow_health:.3f}",
                "value": current.gradient_flow_health,
                "threshold": 0.5
            })

        # Poor gradient balance
        if current.gradient_balance < 0.3:
            anomalies.append({
                "type": "gradient_balance_anomaly",
                "severity": "warning",
                "message": f"Poor gradient balance: {current.gradient_balance:.3f}",
                "value": current.gradient_balance,
                "threshold": 0.3
            })

        # Poor gradient stability
        if current.gradient_stability < 0.3:
            anomalies.append({
                "type": "gradient_stability_anomaly",
                "severity": "warning",
                "message": f"Poor gradient stability: {current.gradient_stability:.3f}",
                "value": current.gradient_stability,
                "threshold": 0.3
            })

        # Strong ratio trend
        if abs(current.ratio_trend) > 0.1:
            anomalies.append({
                "type": "ratio_trend_anomaly",
                "severity": "warning",
                "message": f"Strong ratio trend: {current.ratio_trend:.4f}",
                "value": current.ratio_trend,
                "threshold": 0.1
            })

        # Enhanced detection anomalies
        # Adaptive explosion risk
        if current.adaptive_explosion_risk > 0.3:
            anomalies.append({
                "type": "adaptive_explosion_anomaly",
                "severity": "critical" if current.adaptive_explosion_risk > 0.6 else "warning",
                "message": f"Adaptive explosion risk: {current.adaptive_explosion_risk:.2%}",
                "value": current.adaptive_explosion_risk,
                "threshold": 0.3
            })

        # Statistical outlier risk
        if current.statistical_outlier_risk > 0.4:
            anomalies.append({
                "type": "statistical_outlier_anomaly",
                "severity": "warning",
                "message": f"Statistical outlier risk: {current.statistical_outlier_risk:.2%}",
                "value": current.statistical_outlier_risk,
                "threshold": 0.4
            })

        # Momentum instability risk
        if current.momentum_instability_risk > 0.5:
            anomalies.append({
                "type": "momentum_instability_anomaly",
                "severity": "critical" if current.momentum_instability_risk > 0.8 else "warning",
                "message": f"Momentum instability risk: {current.momentum_instability_risk:.2%}",
                "value": current.momentum_instability_risk,
                "threshold": 0.5
            })

        # Multi-scale anomaly risk
        if current.multi_scale_anomaly_risk > 0.6:
            anomalies.append({
                "type": "multi_scale_anomaly",
                "severity": "warning",
                "message": f"Multi-scale anomaly risk: {current.multi_scale_anomaly_risk:.2%}",
                "value": current.multi_scale_anomaly_risk,
                "threshold": 0.6
            })

        # Early warning system
        if current.early_warning_score > self.early_warning_threshold:
            anomalies.append({
                "type": "early_warning_anomaly",
                "severity": "warning",
                "message": f"Early warning: Gradual deterioration detected ({current.early_warning_score:.2%})",
                "value": current.early_warning_score,
                "threshold": self.early_warning_threshold
            })

        # Change point detection
        if current.change_point_detected:
            anomalies.append({
                "type": "change_point_anomaly",
                "severity": "warning",
                "message": "Sudden change in gradient pattern detected",
                "value": 1.0,
                "threshold": 0.0
            })

        # Gradient acceleration
        if abs(current.gradient_acceleration) > 0.1:
            anomalies.append({
                "type": "gradient_acceleration_anomaly",
                "severity": "warning",
                "message": f"Gradient acceleration: {current.gradient_acceleration:.4f}",
                "value": current.gradient_acceleration,
                "threshold": 0.1
            })

        return anomalies

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of gradient norms analysis."""
        if not self.metrics_history:
            return {}

        return {
            "total_steps": len(self.metrics_history),
            "current_policy_grad_norm": self.current_metrics.policy_grad_norm,
            "current_value_grad_norm": self.current_metrics.value_grad_norm,
            "current_total_grad_norm": self.current_metrics.total_grad_norm,
            "current_policy_value_ratio": self.current_metrics.policy_value_ratio,
            "gradient_health_score": self.current_metrics.gradient_health_score,
            "training_stability": self.current_metrics.training_stability,
            "gradient_flow_health": self.current_metrics.gradient_flow_health,
            "gradient_balance": self.current_metrics.gradient_balance,
            "gradient_stability": self.current_metrics.gradient_stability,
            "exploding_gradient_risk": self.current_metrics.exploding_gradient_risk,
            "vanishing_gradient_risk": self.current_metrics.vanishing_gradient_risk,
            "gradient_imbalance_risk": self.current_metrics.gradient_imbalance_risk,
            "policy_grad_trend": self.current_metrics.policy_grad_trend,
            "value_grad_trend": self.current_metrics.value_grad_trend,
            "ratio_trend": self.current_metrics.ratio_trend,
            # Enhanced detection metrics
            "adaptive_explosion_risk": self.current_metrics.adaptive_explosion_risk,
            "statistical_outlier_risk": self.current_metrics.statistical_outlier_risk,
            "momentum_instability_risk": self.current_metrics.momentum_instability_risk,
            "multi_scale_anomaly_risk": self.current_metrics.multi_scale_anomaly_risk,
            "early_warning_score": self.current_metrics.early_warning_score,
            "exponential_smoothing_trend": self.current_metrics.exponential_smoothing_trend,
            "change_point_detected": self.current_metrics.change_point_detected,
            "gradient_acceleration": self.current_metrics.gradient_acceleration,
            "anomalies": self.get_anomalies()
        }
