"""Comprehensive gradient norms analysis for PPO training."""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import warnings


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
    ):
        """Initialize gradient norms analyzer.
        
        Args:
            window_size: Size of rolling window for analysis
            trend_window: Window size for trend analysis
            exploding_threshold: Threshold for exploding gradient detection
            vanishing_threshold: Threshold for vanishing gradient detection
            imbalance_threshold: Threshold for gradient imbalance detection
        """
        self.window_size = window_size
        self.trend_window = trend_window
        self.exploding_threshold = exploding_threshold
        self.vanishing_threshold = vanishing_threshold
        self.imbalance_threshold = imbalance_threshold
        
        # Data storage
        self.policy_grad_history: deque = deque(maxlen=window_size)
        self.value_grad_history: deque = deque(maxlen=window_size)
        self.total_grad_history: deque = deque(maxlen=window_size)
        self.ratio_history: deque = deque(maxlen=window_size)
        self.step_history: deque = deque(maxlen=window_size)
        
        # Analysis state
        self.current_metrics = GradientNormsMetrics()
        self.metrics_history: List[GradientNormsMetrics] = []
        
        print(f"📊 Gradient Norms Analyzer initialized - "
              f"Exploding: {exploding_threshold}, Vanishing: {vanishing_threshold}, "
              f"Imbalance: {imbalance_threshold}")
    
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
        recent_policy = list(self.policy_grad_history)[-self.trend_window:]
        steps = list(range(len(recent_policy)))
        if len(recent_policy) > 1:
            self.current_metrics.policy_grad_trend = np.polyfit(steps, recent_policy, 1)[0]
        
        # Value gradient trend
        recent_value = list(self.value_grad_history)[-self.trend_window:]
        if len(recent_value) > 1:
            self.current_metrics.value_grad_trend = np.polyfit(steps, recent_value, 1)[0]
        
        # Ratio trend
        recent_ratio = list(self.ratio_history)[-self.trend_window:]
        if len(recent_ratio) > 1:
            self.current_metrics.ratio_trend = np.polyfit(steps, recent_ratio, 1)[0]
    
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
        self.current_metrics.gradient_balance = np.mean(balance_scores)
    
    def _analyze_gradient_stability(self):
        """Analyze gradient stability over time."""
        if len(self.policy_grad_history) < 10:
            return
        
        recent_policy = list(self.policy_grad_history)[-10:]
        recent_value = list(self.value_grad_history)[-10:]
        
        # Calculate coefficient of variation (std/mean) for stability
        policy_cv = np.std(recent_policy) / max(np.mean(recent_policy), 1e-8)
        value_cv = np.std(recent_value) / max(np.mean(recent_value), 1e-8)
        
        # Lower CV is more stable
        policy_stability = max(0, 1 - policy_cv)
        value_stability = max(0, 1 - value_cv)
        
        self.current_metrics.gradient_stability = (policy_stability + value_stability) / 2
    
    def _detect_gradient_anomalies(self):
        """Detect gradient anomalies."""
        if len(self.policy_grad_history) < 5:
            return
        
        recent_policy = list(self.policy_grad_history)[-5:]
        recent_value = list(self.value_grad_history)[-5:]
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
        """Calculate overall health scores."""
        # Gradient health score
        gradient_health = (
            self.current_metrics.gradient_flow_health * 0.3 +
            self.current_metrics.gradient_balance * 0.3 +
            self.current_metrics.gradient_stability * 0.2 +
            max(0, 1 - self.current_metrics.exploding_gradient_risk) * 0.1 +
            max(0, 1 - self.current_metrics.vanishing_gradient_risk) * 0.1
        )
        self.current_metrics.gradient_health_score = max(0, min(1, gradient_health))
        
        # Training stability
        training_stability = (
            self.current_metrics.gradient_stability * 0.4 +
            max(0, 1 - self.current_metrics.gradient_imbalance_risk) * 0.3 +
            max(0, 1 - abs(self.current_metrics.ratio_trend) * 10) * 0.3
        )
        self.current_metrics.training_stability = max(0, min(1, training_stability))
    
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
            "anomalies": self.get_anomalies()
        }