"""
Advanced Anomaly Detection System for RLHF Training.

This module provides sophisticated detection rules for:
- Gradient explosion/vanishing detection
- Learning rate schedule anomalies
- Batch size impact analysis
- Model convergence tracking
- Reward model calibration drift
"""

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy import stats


@dataclass
class AnomalyAlert:
    """Represents an anomaly detection alert."""
    severity: str  # 'low', 'medium', 'high', 'critical'
    category: str  # 'gradient', 'learning_rate', 'batch_size', 'convergence', 'reward_drift'
    message: str
    step: int
    value: float
    threshold: float
    confidence: float = 0.0  # Confidence score (0.0 to 1.0)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GradientStats:
    """Statistics for gradient analysis."""
    mean: float
    std: float
    max: float
    min: float
    norm: float
    explosion_ratio: float
    vanishing_ratio: float


@dataclass
class ConvergenceMetrics:
    """Metrics for convergence tracking."""
    loss_trend: float
    plateau_detected: bool
    convergence_rate: float
    stability_score: float
    improvement_rate: float


class GradientAnomalyDetector:
    """Detects gradient explosion and vanishing problems."""

    def __init__(
        self,
        explosion_threshold: float = 50.0,  # Increased from 10.0 to reduce false positives
        vanishing_threshold: float = 1e-8,  # Increased from 1e-6 to reduce false positives
        window_size: int = 100,
        alert_threshold: float = 2.0  # Reduced from 5.0 to be more specific
    ):
        self.explosion_threshold = explosion_threshold
        self.vanishing_threshold = vanishing_threshold
        self.window_size = window_size
        self.alert_threshold = alert_threshold

        self.gradient_history = deque(maxlen=window_size)
        self.norm_history = deque(maxlen=window_size)
        self.alerts = []

    def analyze_gradients(self, model: nn.Module, step: int) -> List[AnomalyAlert]:
        """Analyze gradients for anomalies."""
        alerts = []

        # Collect gradient statistics
        total_norm = 0.0
        gradient_stats = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                total_norm += grad_norm ** 2

                grad_stats = GradientStats(
                    mean=param.grad.data.mean().item(),
                    std=param.grad.data.std().item(),
                    max=param.grad.data.max().item(),
                    min=param.grad.data.min().item(),
                    norm=grad_norm,
                    explosion_ratio=grad_norm / (param.data.norm(2).item() + 1e-8),
                    vanishing_ratio=grad_norm / (param.data.norm(2).item() + 1e-8)
                )
                gradient_stats.append((name, grad_stats))

        total_norm = total_norm ** 0.5

        # Store in history
        self.gradient_history.append(gradient_stats)
        self.norm_history.append(total_norm)

        # Check for gradient explosion with confidence scoring
        if total_norm > self.explosion_threshold:
            # Calculate confidence based on how extreme the value is
            confidence = min(1.0, (total_norm - self.explosion_threshold) / self.explosion_threshold)
            severity = 'critical' if total_norm > self.explosion_threshold * 3 else 'high'

            # Only alert if confidence is high enough (reduces false positives)
            if confidence > 0.3:  # Only alert for 30%+ above threshold
                # Convert gradient stats to serializable format
                serializable_stats = [(name, stats.__dict__) for name, stats in gradient_stats]
                alerts.append(AnomalyAlert(
                    severity=severity,
                    category='gradient',
                    message=f"Gradient explosion detected: norm={total_norm:.4f} > {self.explosion_threshold} (confidence: {confidence:.2f})",
                    step=step,
                    value=total_norm,
                    threshold=self.explosion_threshold,
                    confidence=confidence,
                    metadata={'gradient_stats': serializable_stats}
                ))

        # Check for gradient vanishing with confidence scoring
        if total_norm < self.vanishing_threshold:
            # Calculate confidence based on how extreme the value is
            confidence = min(1.0, (self.vanishing_threshold - total_norm) / self.vanishing_threshold)

            # Only alert if confidence is high enough and we have enough history
            if confidence > 0.5 and len(self.norm_history) > 20:  # Need more history for vanishing detection
                # Convert gradient stats to serializable format
                serializable_stats = [(name, stats.__dict__) for name, stats in gradient_stats]
                alerts.append(AnomalyAlert(
                    severity='high',
                    category='gradient',
                    message=f"Gradient vanishing detected: norm={total_norm:.8f} < {self.vanishing_threshold} (confidence: {confidence:.2f})",
                    step=step,
                    value=total_norm,
                    threshold=self.vanishing_threshold,
                    confidence=confidence,
                    metadata={'gradient_stats': serializable_stats}
                ))

        # Check for sudden changes in gradient norms with improved logic
        if len(self.norm_history) >= 20:  # Need more history for reliable variance detection
            recent_norms = list(self.norm_history)[-20:]
            norm_std = np.std(recent_norms)
            norm_mean = np.mean(recent_norms)

            # Calculate coefficient of variation
            cv = norm_std / (norm_mean + 1e-8)

            # Only alert if variance is truly excessive and consistent
            if cv > self.alert_threshold and norm_mean > 0.1:  # Only for meaningful gradient magnitudes
                confidence = min(1.0, (cv - self.alert_threshold) / self.alert_threshold)

                # Only alert if confidence is high enough
                if confidence > 0.4:
                    alerts.append(AnomalyAlert(
                        severity='medium',
                        category='gradient',
                        message=f"High gradient variance detected: CV={cv:.4f} > {self.alert_threshold} (confidence: {confidence:.2f})",
                        step=step,
                        value=cv,
                        threshold=self.alert_threshold,
                        confidence=confidence,
                        metadata={'recent_norms': recent_norms, 'norm_mean': norm_mean, 'norm_std': norm_std}
                    ))

        return alerts


class LearningRateAnomalyDetector:
    """Detects anomalies in learning rate schedules."""

    def __init__(
        self,
        change_threshold: float = 0.8,  # Increased from 0.5 to allow normal scheduler behavior
        window_size: int = 50,
        min_lr: float = 1e-10,  # More lenient minimum
        max_lr: float = 10.0,   # More lenient maximum
        consecutive_threshold: int = 3  # Require consecutive changes for alert
    ):
        self.change_threshold = change_threshold
        self.window_size = window_size
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.consecutive_threshold = consecutive_threshold

        self.lr_history = deque(maxlen=window_size)
        self.alerts = []

    def analyze_learning_rate(self, optimizer: torch.optim.Optimizer, step: int) -> List[AnomalyAlert]:
        """Analyze learning rate for anomalies."""
        alerts = []

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        self.lr_history.append(current_lr)

        # Check for out-of-range learning rates with confidence scoring
        if current_lr < self.min_lr:
            confidence = min(1.0, (self.min_lr - current_lr) / self.min_lr)
            if confidence > 0.8:  # Only alert for very low learning rates
                alerts.append(AnomalyAlert(
                    severity='high',
                    category='learning_rate',
                    message=f"Learning rate too low: {current_lr:.2e} < {self.min_lr:.2e} (confidence: {confidence:.2f})",
                    step=step,
                    value=current_lr,
                    threshold=self.min_lr,
                    confidence=confidence,
                    metadata={'optimizer_state': optimizer.state_dict()}
                ))

        if current_lr > self.max_lr:
            confidence = min(1.0, (current_lr - self.max_lr) / self.max_lr)
            if confidence > 0.5:  # Alert for moderately high learning rates
                alerts.append(AnomalyAlert(
                    severity='high',
                    category='learning_rate',
                    message=f"Learning rate too high: {current_lr:.2e} > {self.max_lr:.2e} (confidence: {confidence:.2f})",
                    step=step,
                    value=current_lr,
                    threshold=self.max_lr,
                    confidence=confidence,
                    metadata={'optimizer_state': optimizer.state_dict()}
                ))

        # Check for sudden changes in learning rate with improved logic
        if len(self.lr_history) >= 10:  # Need more history for reliable detection
            recent_lrs = list(self.lr_history)[-10:]
            lr_changes = [abs(recent_lrs[i] - recent_lrs[i-1]) / (recent_lrs[i-1] + 1e-8)
                         for i in range(1, len(recent_lrs))]

            # Count consecutive large changes (more reliable than single large change)
            consecutive_large_changes = 0
            max_consecutive = 0
            for change in lr_changes:
                if change > self.change_threshold:
                    consecutive_large_changes += 1
                    max_consecutive = max(max_consecutive, consecutive_large_changes)
                else:
                    consecutive_large_changes = 0

            # Only alert if we have multiple consecutive large changes
            if max_consecutive >= self.consecutive_threshold:
                confidence = min(1.0, max_consecutive / 5.0)  # Higher confidence for more consecutive changes
                alerts.append(AnomalyAlert(
                    severity='medium',
                    category='learning_rate',
                    message=f"Consecutive learning rate changes detected: {max_consecutive} consecutive changes > {self.change_threshold:.2%} (confidence: {confidence:.2f})",
                    step=step,
                    value=max_consecutive,
                    threshold=self.consecutive_threshold,
                    confidence=confidence,
                    metadata={'recent_lrs': recent_lrs, 'changes': lr_changes, 'max_consecutive': max_consecutive}
                ))

        return alerts


class BatchSizeImpactAnalyzer:
    """Analyzes the impact of batch size changes on training."""

    def __init__(
        self,
        performance_threshold: float = 0.1,
        window_size: int = 20
    ):
        self.performance_threshold = performance_threshold
        self.window_size = window_size

        self.batch_history = deque(maxlen=window_size)
        self.loss_history = deque(maxlen=window_size)
        self.alerts = []

    def analyze_batch_impact(
        self,
        batch_size: int,
        loss: float,
        step: int
    ) -> List[AnomalyAlert]:
        """Analyze batch size impact on training performance."""
        alerts = []

        self.batch_history.append(batch_size)
        self.loss_history.append(loss)

        # Check for batch size changes and their impact
        if len(self.batch_history) >= 10:
            recent_batches = list(self.batch_history)[-10:]
            recent_losses = list(self.loss_history)[-10:]

            # Detect batch size changes
            batch_changes = [recent_batches[i] != recent_batches[i-1]
                           for i in range(1, len(recent_batches))]

            if any(batch_changes):
                # Analyze performance before and after batch size change
                change_idx = next(i for i, changed in enumerate(batch_changes) if changed)

                if change_idx > 2 and change_idx < len(recent_losses) - 2:
                    before_loss = np.mean(recent_losses[:change_idx])
                    after_loss = np.mean(recent_losses[change_idx+1:])

                    performance_change = (after_loss - before_loss) / before_loss

                    if abs(performance_change) > self.performance_threshold:
                        severity = 'high' if abs(performance_change) > self.performance_threshold * 2 else 'medium'
                        alerts.append(AnomalyAlert(
                            severity=severity,
                            category='batch_size',
                            message=f"Batch size change impact: {performance_change:.2%} performance change",
                            step=step,
                            value=performance_change,
                            threshold=self.performance_threshold,
                            metadata={
                                'batch_size_change': recent_batches[change_idx-1] != recent_batches[change_idx],
                                'before_loss': before_loss,
                                'after_loss': after_loss,
                                'change_idx': change_idx
                            }
                        ))

        return alerts


class ConvergenceTracker:
    """Tracks model convergence and detects convergence issues."""

    def __init__(
        self,
        plateau_threshold: float = 0.001,
        plateau_window: int = 50,
        min_improvement: float = 1e-6
    ):
        self.plateau_threshold = plateau_threshold
        self.plateau_window = plateau_window
        self.min_improvement = min_improvement

        self.loss_history = deque(maxlen=plateau_window * 2)
        self.alerts = []

    def analyze_convergence(self, loss: float, step: int) -> Tuple[ConvergenceMetrics, List[AnomalyAlert]]:
        """Analyze convergence metrics and detect issues."""
        alerts = []

        self.loss_history.append(loss)

        if len(self.loss_history) < self.plateau_window:
            return ConvergenceMetrics(0, False, 0, 0, 0), alerts

        recent_losses = list(self.loss_history)[-self.plateau_window:]

        # Calculate loss trend
        x = np.arange(len(recent_losses))
        slope, _, r_value, _, _ = stats.linregress(x, recent_losses)
        loss_trend = slope

        # Detect plateau
        loss_std = np.std(recent_losses)
        plateau_detected = loss_std < self.plateau_threshold

        # Calculate convergence rate
        if len(recent_losses) >= 10:
            early_loss = np.mean(recent_losses[:5])
            late_loss = np.mean(recent_losses[-5:])
            convergence_rate = (early_loss - late_loss) / early_loss
        else:
            convergence_rate = 0

        # Calculate stability score
        stability_score = 1.0 / (1.0 + loss_std)

        # Calculate improvement rate
        if len(self.loss_history) >= 20:
            old_losses = list(self.loss_history)[-40:-20]
            new_losses = list(self.loss_history)[-20:]
            old_avg = np.mean(old_losses)
            new_avg = np.mean(new_losses)
            improvement_rate = (old_avg - new_avg) / old_avg
        else:
            improvement_rate = 0

        metrics = ConvergenceMetrics(
            loss_trend=loss_trend,
            plateau_detected=plateau_detected,
            convergence_rate=convergence_rate,
            stability_score=stability_score,
            improvement_rate=improvement_rate
        )

        # Generate alerts
        if plateau_detected and abs(improvement_rate) < self.min_improvement:
            alerts.append(AnomalyAlert(
                severity='medium',
                category='convergence',
                message=f"Training plateau detected: loss_std={loss_std:.6f} < {self.plateau_threshold}",
                step=step,
                value=loss_std,
                threshold=self.plateau_threshold,
                metadata={'metrics': metrics}
            ))

        if convergence_rate < -0.1:  # Loss is increasing
            alerts.append(AnomalyAlert(
                severity='high',
                category='convergence',
                message=f"Loss increasing: convergence_rate={convergence_rate:.4f}",
                step=step,
                value=convergence_rate,
                threshold=-0.1,
                metadata={'metrics': metrics}
            ))

        return metrics, alerts


class RewardCalibrationDriftDetector:
    """Enhanced reward model calibration drift detection."""

    def __init__(
        self,
        drift_threshold: float = 0.3,  # Increased from 0.1 to reduce false positives
        calibration_threshold: float = 0.5,  # More lenient calibration threshold
        window_size: int = 100,
        min_samples: int = 20  # Require more samples for reliable detection
    ):
        self.drift_threshold = drift_threshold
        self.calibration_threshold = calibration_threshold
        self.window_size = window_size
        self.min_samples = min_samples

        self.reward_history = deque(maxlen=window_size)
        self.calibration_history = deque(maxlen=window_size)
        self.alerts = []

    def analyze_reward_drift(
        self,
        rewards: np.ndarray,
        predictions: np.ndarray,
        step: int
    ) -> List[AnomalyAlert]:
        """Analyze reward model calibration drift."""
        alerts = []

        # Calculate calibration metrics
        calibration_score = self._calculate_calibration_score(rewards, predictions)
        self.calibration_history.append(calibration_score)

        # Store reward statistics
        reward_stats = {
            'mean': np.mean(rewards),
            'std': np.std(rewards),
            'min': np.min(rewards),
            'max': np.max(rewards)
        }
        self.reward_history.append(reward_stats)

        # Check calibration quality with confidence scoring
        if calibration_score < self.calibration_threshold:
            confidence = min(1.0, (self.calibration_threshold - calibration_score) / self.calibration_threshold)
            if confidence > 0.6:  # Only alert for significantly poor calibration
                alerts.append(AnomalyAlert(
                    severity='high',
                    category='reward_drift',
                    message=f"Poor calibration detected: score={calibration_score:.4f} < {self.calibration_threshold} (confidence: {confidence:.2f})",
                    step=step,
                    value=calibration_score,
                    threshold=self.calibration_threshold,
                    confidence=confidence,
                    metadata={'rewards': rewards, 'predictions': predictions}
                ))

        # Check for calibration drift with improved logic
        if len(self.calibration_history) >= self.min_samples:
            recent_calibration = list(self.calibration_history)[-self.min_samples:]
            calibration_trend = np.polyfit(range(len(recent_calibration)), recent_calibration, 1)[0]

            # Calculate confidence based on trend strength and consistency
            trend_strength = abs(calibration_trend)
            confidence = min(1.0, trend_strength / self.drift_threshold)

            # Only alert if trend is strong and consistent
            if trend_strength > self.drift_threshold and confidence > 0.7:
                alerts.append(AnomalyAlert(
                    severity='medium',
                    category='reward_drift',
                    message=f"Calibration drift detected: trend={calibration_trend:.4f} (confidence: {confidence:.2f})",
                    step=step,
                    value=calibration_trend,
                    threshold=self.drift_threshold,
                    confidence=confidence,
                    metadata={'recent_calibration': recent_calibration, 'trend_strength': trend_strength}
                ))

        # Check for reward distribution changes with improved logic
        if len(self.reward_history) >= 20:  # Need more history for reliable detection
            recent_rewards = list(self.reward_history)[-10:]
            old_rewards = list(self.reward_history)[-20:-10]

            if old_rewards:
                old_mean = np.mean([r['mean'] for r in old_rewards])
                new_mean = np.mean([r['mean'] for r in recent_rewards])

                mean_change = abs(new_mean - old_mean) / (old_mean + 1e-8)

                # Calculate confidence based on change magnitude and consistency
                confidence = min(1.0, mean_change / self.drift_threshold)

                # Only alert if change is significant and consistent
                if mean_change > self.drift_threshold and confidence > 0.6:
                    alerts.append(AnomalyAlert(
                        severity='medium',
                        category='reward_drift',
                        message=f"Reward distribution drift: mean change={mean_change:.4f} (confidence: {confidence:.2f})",
                        step=step,
                        value=mean_change,
                        threshold=self.drift_threshold,
                        confidence=confidence,
                        metadata={'old_mean': old_mean, 'new_mean': new_mean, 'change_ratio': mean_change}
                    ))

        return alerts

    def _calculate_calibration_score(self, rewards: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate calibration score using reliability diagram."""
        try:
            # Bin the predictions and calculate calibration
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]

            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
                prop_in_bin = in_bin.mean()

                if prop_in_bin > 0:
                    accuracy_in_bin = rewards[in_bin].mean()
                    avg_confidence_in_bin = predictions[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            return 1.0 - ece  # Higher is better
        except (ValueError, RuntimeError) as e:
            # Log the specific error for debugging
            print(f"Warning: Error calculating calibration score: {e}")
            return 0.5  # Default neutral score


class AdvancedAnomalyDetector:
    """Main anomaly detection system that coordinates all detectors."""

    def __init__(
        self,
        output_dir: str = "anomaly_detection",
        save_alerts: bool = True,
        **kwargs
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_alerts = save_alerts

        # Initialize detectors with improved default thresholds
        gradient_config = {
            'explosion_threshold': 50.0,  # More lenient
            'vanishing_threshold': 1e-8,  # More lenient
            'alert_threshold': 2.0,       # More specific
            **kwargs.get('gradient', {})
        }

        lr_config = {
            'change_threshold': 0.8,      # Allow normal scheduler behavior
            'min_lr': 1e-10,             # More lenient
            'max_lr': 10.0,              # More lenient
            'consecutive_threshold': 3,   # Require consecutive changes
            **kwargs.get('learning_rate', {})
        }

        reward_config = {
            'drift_threshold': 0.3,       # More lenient
            'calibration_threshold': 0.5, # More lenient
            'min_samples': 20,            # Require more samples
            **kwargs.get('reward_drift', {})
        }

        self.gradient_detector = GradientAnomalyDetector(**gradient_config)
        self.lr_detector = LearningRateAnomalyDetector(**lr_config)
        self.batch_analyzer = BatchSizeImpactAnalyzer(**kwargs.get('batch_size', {}))
        self.convergence_tracker = ConvergenceTracker(**kwargs.get('convergence', {}))
        self.reward_detector = RewardCalibrationDriftDetector(**reward_config)

        self.all_alerts = []
        self.step_count = 0

    def analyze_training_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: float,
        batch_size: int,
        rewards: Optional[np.ndarray] = None,
        predictions: Optional[np.ndarray] = None
    ) -> List[AnomalyAlert]:
        """Analyze a single training step for anomalies."""
        self.step_count += 1
        alerts = []

        # Gradient analysis
        gradient_alerts = self.gradient_detector.analyze_gradients(model, self.step_count)
        alerts.extend(gradient_alerts)

        # Learning rate analysis
        lr_alerts = self.lr_detector.analyze_learning_rate(optimizer, self.step_count)
        alerts.extend(lr_alerts)

        # Batch size impact analysis
        batch_alerts = self.batch_analyzer.analyze_batch_impact(batch_size, loss, self.step_count)
        alerts.extend(batch_alerts)

        # Convergence tracking
        convergence_metrics, convergence_alerts = self.convergence_tracker.analyze_convergence(loss, self.step_count)
        alerts.extend(convergence_alerts)

        # Reward calibration drift (if available)
        if rewards is not None and predictions is not None:
            reward_alerts = self.reward_detector.analyze_reward_drift(rewards, predictions, self.step_count)
            alerts.extend(reward_alerts)

        # Store alerts
        self.all_alerts.extend(alerts)

        # Save alerts if enabled
        if self.save_alerts and alerts:
            self._save_alerts(alerts)

        return alerts

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all detected anomalies."""
        if not self.all_alerts:
            return {"total_alerts": 0, "by_category": {}, "by_severity": {}}

        by_category = {}
        by_severity = {}

        for alert in self.all_alerts:
            by_category[alert.category] = by_category.get(alert.category, 0) + 1
            by_severity[alert.severity] = by_severity.get(alert.severity, 0) + 1

        return {
            "total_alerts": len(self.all_alerts),
            "by_category": by_category,
            "by_severity": by_severity,
            "latest_alerts": self.all_alerts[-10:] if len(self.all_alerts) > 10 else self.all_alerts
        }

    def _save_alerts(self, alerts: List[AnomalyAlert]):
        """Save alerts to file."""
        alert_data = []
        for alert in alerts:
            # Convert metadata to JSON-serializable format
            serializable_metadata = self._make_serializable(alert.metadata)

            alert_data.append({
                "severity": alert.severity,
                "category": alert.category,
                "message": alert.message,
                "step": alert.step,
                "value": alert.value,
                "threshold": alert.threshold,
                "metadata": serializable_metadata
            })

        alert_file = self.output_dir / f"alerts_step_{self.step_count}.json"
        with open(alert_file, 'w') as f:
            json.dump(alert_data, f, indent=2)

    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (GradientStats, ConvergenceMetrics)):
            return self._make_serializable(obj.__dict__)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, (bool, int, float, str)):
            return obj
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            try:
                # Try to convert to string as last resort
                return str(obj)
            except Exception as e:
                print(f"Warning: Could not serialize object: {e}")
                return None

    def save_final_report(self):
        """Save final anomaly detection report."""
        summary = self.get_summary()

        # Make summary serializable
        serializable_summary = self._make_serializable(summary)

        report_file = self.output_dir / "anomaly_detection_report.json"
        with open(report_file, 'w') as f:
            json.dump(serializable_summary, f, indent=2, default=str)

        print(f"Anomaly detection report saved to {report_file}")
        return report_file
